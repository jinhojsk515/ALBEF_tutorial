import argparse
from pathlib import Path
import utils
import torch
import numpy as np
import random
from pretrain_albef import ALBEF
import time
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer
import datetime
from torchvision import transforms
import torchvision.transforms as T
from dataset import MIMIC_CXRDataset
from torch.utils.data import DataLoader
from vit import interpolate_pos_embed
import torch.nn.functional as F

# ALBEF's strategy for retrieval task(ex: Image->Text)
# When an input image is given, the model choose 'k-test' number of texts for that image,
# according to their cosine similarity of their contrastive-loss embedding vectors.(this removes a lot of irrelavant images)
# Then, the model takes those 'k-test' texts and the input image to perform ITM.
# 'k-test' texts are ranked according to their ITM probability of being "matched", and become the retrieval output.

# For this code, "Correct retrieval" is defined as "the image/text with the same diagnosis labels"


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device, config):
    # test
    model.eval()
    print('Computing features for evaluation...')
    start_time = time.time()

    texts=[l['text'] for l in data_loader.dataset.data]
    num_text = len(texts)
    text_bs = 256
    text_feats = []
    text_embeds = []
    text_atts = []
    for i in range(0, num_text, text_bs):
        text = texts[i: min(num_text, i + text_bs)]
        text_input = tokenizer(text, padding='max_length', truncation=True, max_length=200, return_tensors="pt").to(
            device)
        text_feat = model.text_encoder.bert(text_input.input_ids, attention_mask=text_input.attention_mask, mode='text').last_hidden_state
        text_embed = F.normalize(model.text_proj(text_feat[:, 0, :]))
        text_embeds.append(text_embed)
        text_feats.append(text_feat)
        text_atts.append(text_input.attention_mask)
    text_embeds = torch.cat(text_embeds, dim=0)     #batch*feature
    text_feats = torch.cat(text_feats, dim=0)       #batch*feature_contrastive
    text_atts = torch.cat(text_atts, dim=0)

    image_feats = []
    image_embeds = []
    labels=[]       #contains the labels(=the criteria for correct retrieval)
    for image, _, label in data_loader:
        image = image.to(device)
        image_feat = model.visual_encoder(image)[:, 0, :]
        image_embed = model.vision_proj(image_feat)
        image_embed = F.normalize(image_embed, dim=-1)

        image_feats.append(image_feat)
        image_embeds.append(image_embed)

        labels.append(label)
    labels=torch.cat(labels,dim=0)

    image_feats = torch.cat(image_feats, dim=0)     #batch*feature_contrastive
    image_embeds = torch.cat(image_embeds, dim=0)   #data*feature

    sims_matrix = image_embeds @ text_embeds.t()    #data*data
    score_matrix_i2t = torch.full((len(data_loader.dataset.data), len(texts)), -100.0).to(device)

    num_tasks = utils.get_world_size()
    rank = utils.get_rank()
    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(sims_matrix[start:end]):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[start + i].repeat(config['k_test'], 1, 1)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output_image = model.text_encoder.bert(encoder_embeds=encoder_output,
                                          attention_mask=encoder_att,
                                          encoder_hidden_states=text_feats[topk_idx],
                                          encoder_attention_mask=text_atts[topk_idx],
                                          return_dict=True,
                                          mode='fusion',
                                          ).last_hidden_state[:, 0, :]
        output_text = model.text_encoder.bert(encoder_embeds=text_feats[topk_idx],
                                         attention_mask=text_atts[topk_idx],
                                         encoder_hidden_states=encoder_output,
                                         encoder_attention_mask=encoder_att,
                                         return_dict=True,
                                         mode='fusion'
                                         ).last_hidden_state[:, 0, :]
        output = torch.cat([output_image, output_text], dim=-1)
        score = model.itm_head(output)[:, 1]
        score_matrix_i2t[start + i, topk_idx] = score

    sims_matrix = sims_matrix.t()
    score_matrix_t2i = torch.full((len(texts), len(data_loader.dataset.data)), -100.0).to(device)

    step = sims_matrix.size(0) // num_tasks + 1
    start = rank * step
    end = min(sims_matrix.size(0), start + step)

    for i, sims in enumerate(sims_matrix[start:end]):
        topk_sim, topk_idx = sims.topk(k=config['k_test'], dim=0)
        encoder_output = image_feats[topk_idx].unsqueeze(1)
        encoder_att = torch.ones(encoder_output.size()[:-1], dtype=torch.long).to(device)
        output_image = model.text_encoder.bert(encoder_embeds=encoder_output,
                                          attention_mask=encoder_att,
                                          encoder_hidden_states=text_feats[start + i].repeat(config['k_test'], 1, 1),
                                          encoder_attention_mask=text_atts[start + i].repeat(config['k_test'], 1),
                                          return_dict=True,
                                          mode='fusion',
                                          ).last_hidden_state[:, 0, :]
        output_text = model.text_encoder.bert(encoder_embeds=text_feats[start + i].repeat(config['k_test'], 1, 1),
                                         attention_mask=text_atts[start + i].repeat(config['k_test'], 1),
                                         encoder_hidden_states=encoder_output,
                                         encoder_attention_mask=encoder_att,
                                         return_dict=True,
                                         mode='fusion'
                                         ).last_hidden_state[:, 0, :]
        output = torch.cat([output_image, output_text], dim=-1)
        score = model.itm_head(output)[:, 1]
        score_matrix_t2i[start + i, topk_idx] = score

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Evaluation time {}'.format(total_time_str))

    return score_matrix_i2t.cpu().numpy(), score_matrix_t2i.cpu().numpy(), labels


@torch.no_grad()
def itm_eval(scores_i2t, scores_t2i, labels):
    # Images->Text
    ranks = np.zeros(scores_i2t.shape[0])
    for index, score in enumerate(scores_i2t):
        inds = np.argsort(score)[::-1]
        # Score
        for num,i in enumerate(inds):
            if torch.sum((labels[i]-labels[index])**2)==0:
                ranks[index]=num
                break
    print('I2T MRR:',np.mean(1/(ranks+1)))
    # Compute metrics
    tr1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    tr5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    tr10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    # Text->Images
    ranks = np.zeros(scores_t2i.shape[0])
    for index, score in enumerate(scores_t2i):
        inds = np.argsort(score)[::-1]

        for num,i in enumerate(inds):
            if torch.sum((labels[i]-labels[index])**2)==0:
                ranks[index]=num
                break
    print('T2I MRR:', np.mean(1/(ranks+1)))

    # Compute metrics
    ir1 = 100.0 * len(np.where(ranks < 1)[0]) / len(ranks)
    ir5 = 100.0 * len(np.where(ranks < 5)[0]) / len(ranks)
    ir10 = 100.0 * len(np.where(ranks < 10)[0]) / len(ranks)

    tr_mean = (tr1 + tr5 + tr10) / 3
    ir_mean = (ir1 + ir5 + ir10) / 3
    r_mean = (tr_mean + ir_mean) / 2

    eval_result = {'i2t_h1': tr1,
                   'i2t_h5': tr5,
                   'i2t_h10': tr10,
                   'i2t_h_mean': tr_mean,
                   't2i_h1': ir1,
                   't2i_h5': ir5,
                   't2i_h10': ir10,
                   't2i_h_mean': ir_mean,
                   'mean': r_mean}
    print('eval result:', eval_result)


def main(args, config):
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    #### Dataset ####
    print("Creating dataset")
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    test_transform = transforms.Compose([
            transforms.Resize((config['image_res'],config['image_res']), interpolation=T.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
    dataset_test = MIMIC_CXRDataset("./data/MIMIC_CXR/Test.jsonl", transform=test_transform, data_length=100)
    test_loader = DataLoader(dataset_test, batch_size=config['batch_size_test'], pin_memory=True, drop_last=False)

    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = ALBEF(config=config, tokenizer=tokenizer)

    if args.checkpoint:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
        m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'], model.visual_encoder_m)
        state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model = model.to(device)
    model_without_ddp = model

    start_time = time.time()
    score_test_i2t, score_test_t2i, labels_test = evaluate(model_without_ddp, test_loader, tokenizer, device, config)
    print('TEST')
    itm_eval(score_test_i2t, score_test_t2i,labels_test)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    ret_config = {
        'image_res': 256,
        'batch_size_train': 4,
        'batch_size_test': 4,
        'alpha': 0.4,
        'queue_size': 2048,    #65536
        'momentum': 0.995,
        'vision_width': 768,
        'embed_dim': 256,
        'temp': 0.07,
        'mlm_probability': 0.15,
        'k_test': 32,
        'distill':True,
        'warm_up':True,
        'bert_config': './config_bert.json',
    }

    main(args, ret_config)
