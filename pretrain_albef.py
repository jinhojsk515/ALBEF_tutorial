from functools import partial
from vit import VisionTransformer, interpolate_pos_embed
# ALBEF authors modified the original huggingface BERT model,
# to enable mixing self-attention layer and cross-attention layer without causal mask.
# To use this, import "xbert"
from xbert import BertConfig, BertForMaskedLM
import torch
import torch.nn.functional as F
from torch import nn


class ALBEF(nn.Module):
    def __init__(self,
                 text_encoder_name='bert-base-uncased',
                 tokenizer=None,
                 config=None,
                 temp=0.07,
                 init_vit=True
                 ):
        super().__init__()

        self.tokenizer = tokenizer
        self.mlm_probability = config['mlm_probability']
        embed_dim = config['embed_dim']

        self.visual_encoder = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=config['vision_width'], depth=6, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6)) #I reduced the model size(depth=6)
        #self.visual_encoder = VisionTransformer(
        #    img_size=config['image_res'], patch_size=16, embed_dim=config['vision_width'], depth=12, num_heads=12,
        #    mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))

        if init_vit:
            print('init_ViT')
            checkpoint = torch.hub.load_state_dict_from_url(
                url="https://dl.fbaipublicfiles.com/deit/deit_base_patch16_224-b5f2ef4d.pth",
                map_location="cpu", check_hash=True)
            state_dict = checkpoint["model"]
            pos_embed_reshaped = interpolate_pos_embed(state_dict['pos_embed'], self.visual_encoder)
            state_dict['pos_embed'] = pos_embed_reshaped
            msg = self.visual_encoder.load_state_dict(state_dict, strict=False)
            print('ViT parameter loading done: ',msg)

        vision_width = config['vision_width']

        bert_config = BertConfig.from_json_file(config['bert_config'])

        # BertForMaskedLM = self.bert(BertModel) + self.cls(BertOnlyMLMHead), BertModel = BertEmbedding + BertLayers
        # This text_encoder's first half layers of the BertLayers are the text-encoder for text data.
        # The other half are multi-modal encoders, which contains cross-attention layers.
        # To use text-encoder only, use the "mode" of "text". To use the multi-modal layer part, use the "mode" of "fusion".
        # If you input nothing for "mode", you will use all layers.
        self.text_encoder = BertForMaskedLM.from_pretrained(text_encoder_name, config=bert_config)
        #self.text_encoder = BertForMaskedLM(config=bert_config)

        text_width = self.text_encoder.config.hidden_size
        self.vision_proj = nn.Linear(vision_width, embed_dim)
        self.text_proj = nn.Linear(text_width, embed_dim)

        self.temp = nn.Parameter(torch.ones([]) * config['temp'])
        self.queue_size = config['queue_size']
        self.momentum = config['momentum']
        self.itm_head = nn.Linear(text_width, 2)

        # create momentum models

        self.visual_encoder_m = VisionTransformer(
            img_size=config['image_res'], patch_size=16, embed_dim=config['vision_width'], depth=6, num_heads=12,
            mlp_ratio=4, qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6))
        self.vision_proj_m = nn.Linear(vision_width, embed_dim)
        self.text_encoder_m = BertForMaskedLM.from_pretrained(text_encoder_name, config=bert_config)
        #self.text_encoder_m = BertForMaskedLM(config=bert_config)

        self.text_proj_m = nn.Linear(text_width, embed_dim)

        self.model_pairs = [[self.visual_encoder, self.visual_encoder_m],
                            [self.vision_proj, self.vision_proj_m],
                            [self.text_encoder, self.text_encoder_m],
                            [self.text_proj, self.text_proj_m],
                            ]

        self.copy_params()

        # create the queue
        self.register_buffer("image_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("text_queue", torch.randn(embed_dim, self.queue_size))
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

        self.image_queue = nn.functional.normalize(self.image_queue, dim=0)
        self.text_queue = nn.functional.normalize(self.text_queue, dim=0)





    def forward(self, image, text, alpha=0):

        with torch.no_grad():
            self.temp.clamp_(0.001, 0.5)

        # Encode image with visual_encoder.
        image_embeds = self.visual_encoder(image)
        image_atts = torch.ones(image_embeds.size()[:-1], dtype=torch.long).to(image.device)    #Build a image's padding mask for BERT input. Unlike text, every image has the same length(1+num_patch), so there's no padding mask. Every entry is 1.
        #Encode text with text_encoder.
        text_embeds = self.text_encoder.bert(text.input_ids, attention_mask=text.attention_mask,
                                             return_dict=True, mode='text').last_hidden_state   #mode='text' to use the text_encoder(=first half)

        #To use in the contrastive loss, project the feature of the cls_token.
        image_feat = F.normalize(self.vision_proj(image_embeds[:, 0, :]), dim=-1)
        text_feat = F.normalize(self.text_proj(text_embeds[:, 0, :]), dim=-1)

        #=================1. CONTRASTIVE LOSS=================#
        # get momentum features. Momentum model is used in contrastive loss & MLM loss.
        with torch.no_grad():
            self._momentum_update() #update the momentum model with current model parameters.

            image_embeds_m = self.visual_encoder_m(image)
            image_feat_m = F.normalize(self.vision_proj_m(image_embeds_m[:, 0, :]), dim=-1)
            image_feat_all = torch.cat([image_feat_m.t(), self.image_queue.clone().detach()], dim=1)    #ALBEF uses a queue to save the old image cls_token features, and bring them back as a negative instance in contrastive loss.

            text_output_m = self.text_encoder_m.bert(text.input_ids, attention_mask=text.attention_mask,return_dict=True, mode='text')
            text_feat_m = F.normalize(self.text_proj_m(text_output_m.last_hidden_state[:, 0, :]), dim=-1)
            text_feat_all = torch.cat([text_feat_m.t(), self.text_queue.clone().detach()], dim=1)   #ALBEF uses a queue to save the old text cls_token features, and bring them back as a negative instance in contrastive loss.

            sim_i2t_m = image_feat_m @ text_feat_all / self.temp
            sim_t2i_m = text_feat_m @ image_feat_all / self.temp

            #Making the target for contrastive loss.
            sim_targets = torch.zeros(sim_i2t_m.size()).to(image.device)
            sim_targets.fill_diagonal_(1)   #This "sim_targets" should be the target for naive contrastive loss, but we mix the output of the momentum model.
            sim_i2t_targets = alpha * F.softmax(sim_i2t_m, dim=1) + (1 - alpha) * sim_targets
            sim_t2i_targets = alpha * F.softmax(sim_t2i_m, dim=1) + (1 - alpha) * sim_targets

        sim_i2t = image_feat @ text_feat_all / self.temp
        sim_t2i = text_feat @ image_feat_all / self.temp

        #cross-entropy loss
        loss_i2t = -torch.sum(F.log_softmax(sim_i2t, dim=1) * sim_i2t_targets, dim=1).mean()
        loss_t2i = -torch.sum(F.log_softmax(sim_t2i, dim=1) * sim_t2i_targets, dim=1).mean()

        loss_ita = (loss_i2t + loss_t2i) / 2
        self._dequeue_and_enqueue(image_feat_m, text_feat_m)


        #=================2. Image-Text Matching loss=================#
        # forward the positve image-text pair

        output_pos = self.text_encoder.bert(encoder_embeds = text_embeds,   #use text_encoder'.bert' only to not use MLM head.
                                        attention_mask = text.attention_mask,
                                        encoder_hidden_states = image_embeds,
                                        encoder_attention_mask = image_atts,
                                        return_dict = True,
                                        mode = 'fusion',    #mode='fusion' to use the multi-modal encoder(=last half)
                                       )
        with torch.no_grad():
            bs = image.size(0)
            weights_i2t = F.softmax(sim_i2t[:, :bs], dim=1)
            weights_t2i = F.softmax(sim_t2i[:, :bs], dim=1)

            weights_i2t.fill_diagonal_(0)
            weights_t2i.fill_diagonal_(0)

        # select a negative image for each text. For hard negative-sampling, choose the instance with the best similarity.
        image_embeds_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_t2i[b], 1).item()
            image_embeds_neg.append(image_embeds[neg_idx])
        image_embeds_neg = torch.stack(image_embeds_neg, dim=0)

        # select a negative text for each image. For hard negative-sampling, choose the instance with the best similarity.
        text_embeds_neg = []
        text_atts_neg = []
        for b in range(bs):
            neg_idx = torch.multinomial(weights_i2t[b], 1).item()
            text_embeds_neg.append(text_embeds[neg_idx])
            text_atts_neg.append(text.attention_mask[neg_idx])
        text_embeds_neg = torch.stack(text_embeds_neg, dim=0)
        text_atts_neg = torch.stack(text_atts_neg, dim=0)

        #text_embeds_all: Original text + Negative text sample for Original image
        text_embeds_all = torch.cat([text_embeds, text_embeds_neg], dim=0)
        text_atts_all = torch.cat([text.attention_mask, text_atts_neg], dim=0)
        #image_embeds_all: Negative image sample for Original text + Original image
        image_embeds_all = torch.cat([image_embeds_neg, image_embeds], dim=0)
        image_atts_all = torch.cat([image_atts, image_atts], dim=0)

        output_neg = self.text_encoder.bert(encoder_embeds = text_embeds_all,
                                        attention_mask = text_atts_all,
                                        encoder_hidden_states = image_embeds_all,
                                        encoder_attention_mask = image_atts_all,
                                        return_dict = True,
                                        mode = 'fusion',
                                       )

        vl_embeddings = torch.cat([output_pos.last_hidden_state[:, 0, :], output_neg.last_hidden_state[:, 0, :]], dim=0)
        vl_output = self.itm_head(vl_embeddings)

        itm_labels = torch.cat([torch.ones(bs, dtype=torch.long), torch.zeros(2 * bs, dtype=torch.long)],
                               dim=0).to(image.device)  #1 for positive pairs, 0 for negative pairs
        loss_itm = F.cross_entropy(vl_output, itm_labels)


        #=================3. MLM loss=================#
        #MLM loss, the same loss used in the original BERT paper.
        input_ids = text.input_ids.clone()
        labels = input_ids.clone()

        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        input_ids, labels = self.mask(input_ids, self.text_encoder.config.vocab_size, image.device, targets=labels,
                                      probability_matrix = probability_matrix)  #"labels" has the value of -100 if that position is not masked in input_ids

        with torch.no_grad():
            logits_m = self.text_encoder_m(input_ids,      #use MLM head this time
                                           attention_mask = text.attention_mask,
                                           encoder_hidden_states = image_embeds_m,
                                           encoder_attention_mask = image_atts,
                                           return_dict = True,
                                           return_logits = True,    #no "mode" = use all layers
                                          )
        mlm_output = self.text_encoder(input_ids,
                                       attention_mask = text.attention_mask,
                                       encoder_hidden_states = image_embeds,
                                       encoder_attention_mask = image_atts,
                                       return_dict = True,
                                       return_logits=True
                                      )
        loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
        loss_mlm = loss_fct(mlm_output.permute((0,2,1)), labels)

        loss_distill = -torch.sum(F.log_softmax(mlm_output, dim=-1) * F.softmax(logits_m,dim=-1), dim=-1)
        loss_distill = loss_distill[labels != -100].mean()
        loss_mlm = (1 - alpha) * loss_mlm + alpha * loss_distill

        return loss_mlm, loss_ita, loss_itm




    @torch.no_grad()
    def copy_params(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data.copy_(param.data)  # initialize
                param_m.requires_grad = False  # not update by gradient

    @torch.no_grad()
    def _momentum_update(self):
        for model_pair in self.model_pairs:
            for param, param_m in zip(model_pair[0].parameters(), model_pair[1].parameters()):
                param_m.data = param_m.data * self.momentum + param.data * (1. - self.momentum)

    @torch.no_grad()
    def _dequeue_and_enqueue(self, image_feat, text_feat):
        # gather keys before updating queue
        #image_feats = concat_all_gather(image_feat)    #This function is needed in multi-gpu training, maybe...
        #text_feats = concat_all_gather(text_feat)
        image_feats = image_feat
        text_feats = text_feat

        batch_size = image_feats.shape[0]

        ptr = int(self.queue_ptr)
        assert self.queue_size % batch_size == 0  # for simplicity

        # replace the keys at ptr (dequeue and enqueue)
        self.image_queue[:, ptr:ptr + batch_size] = image_feats.T
        self.text_queue[:, ptr:ptr + batch_size] = text_feats.T
        ptr = (ptr + batch_size) % self.queue_size  # move pointer

        self.queue_ptr[0] = ptr

    def mask(self, input_ids, vocab_size, device, targets=None, masked_indices=None, probability_matrix=None):
        if masked_indices is None:
            masked_indices = torch.bernoulli(probability_matrix).bool()

        masked_indices[input_ids == self.tokenizer.pad_token_id] = False
        masked_indices[input_ids == self.tokenizer.cls_token_id] = False

        if targets is not None:
            targets[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = torch.bernoulli(torch.full(input_ids.shape, 0.8)).bool() & masked_indices
        input_ids[indices_replaced] = self.tokenizer.mask_token_id

        # 10% of the time, we replace masked input tokens with random word
        indices_random = torch.bernoulli(torch.full(input_ids.shape, 0.5)).bool() & masked_indices & ~indices_replaced
        random_words = torch.randint(vocab_size, input_ids.shape, dtype=torch.long).to(device)
        input_ids[indices_random] = random_words[indices_random]
        # The rest of the time (10% of the time) we keep the masked input tokens unchanged

        if targets is not None:
            return input_ids, targets
        else:
            return input_ids


@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
                      for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output