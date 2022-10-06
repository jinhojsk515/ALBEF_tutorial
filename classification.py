import argparse
from tqdm import tqdm
from pathlib import Path
import utils
import torch
import numpy as np
import random
from classification_albef import ALBEF
import time
import os
from vit import interpolate_pos_embed
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer
import datetime
from torchvision import transforms
import torchvision.transforms as T
from dataset import MIMIC_CXRDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from scheduler import create_scheduler
from sklearn.metrics import f1_score,roc_auc_score


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()

    header = 'Train Epoch: [{}]'.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    tqdm_data_loader = tqdm(data_loader, miniters=print_freq, desc=header)
    for i, (images, text, targets) in enumerate(tqdm_data_loader):

        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)
        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)

        if epoch > 0 or not config['warm_up']:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1., i / len(data_loader))

        loss = model(images, text_inputs, targets=targets, train=True, alpha=alpha)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        tqdm_data_loader.set_description(f'loss={loss.item():.4f}, lr={optimizer.param_groups[0]["lr"]:.6f}')

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)


@torch.no_grad()
def evaluate(model, data_loader, tokenizer, device):
    # test
    model.eval()

    preds=[]
    answers=[]
    for images, text, targets in data_loader:
        images, targets = images.to(device, non_blocking=True), targets.to(device, non_blocking=True)

        text_inputs = tokenizer(text, padding='longest', return_tensors="pt").to(device)

        prediction = model(images, text_inputs, targets=targets, train=False)

        prediction=torch.sigmoid(prediction)
        prediction[prediction>=0.5]=1.
        prediction[prediction<0.5]=0.
        preds.append(prediction.cpu())
        answers.append(targets.cpu())

    preds=torch.cat(preds,dim=0)
    answers=torch.cat(answers,dim=0)
    print('F1 score[micro]:',f1_score(answers, preds, average='micro'))
    print('AUROC:',roc_auc_score(answers, preds,average='micro'))
    print('Accuracy:',((answers == preds).sum() / answers.numel()).item())
    score = f1_score(answers, preds, average='micro')

    return score


def main(args, config):
    print('aa', args.evaluate)
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
    train_transform = transforms.Compose([
            transforms.RandomResizedCrop(config['image_res'],scale=(0.5, 1.0),interpolation=T.InterpolationMode.BICUBIC),
            #RandomAugment(2,7,isPIL=True,augs=['Identity','AutoContrast','Equalize','Brightness','Sharpness',
            #                                  'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
            transforms.ToTensor(),
            normalize,
        ])
    test_transform = transforms.Compose([
            transforms.Resize((config['image_res'],config['image_res']),interpolation=T.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            normalize,
        ])
    dataset_train = MIMIC_CXRDataset("./data/MIMIC_CXR/Train.jsonl", transform=train_transform)
    dataset_val = MIMIC_CXRDataset("./data/MIMIC_CXR/Valid.jsonl", transform=test_transform)
    dataset_test = MIMIC_CXRDataset("./data/MIMIC_CXR/Test.jsonl", transform=test_transform)
    train_loader = DataLoader(dataset_train, batch_size=config['batch_size_train'], pin_memory=True, drop_last=True)
    val_loader = DataLoader(dataset_val, batch_size=config['batch_size_test'], pin_memory=True, drop_last=False)
    test_loader = DataLoader(dataset_test, batch_size=config['batch_size_test'], pin_memory=True, drop_last=False)


    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = ALBEF(config=config, text_encoder=args.text_encoder, tokenizer=tokenizer)

    if args.checkpoint:
        print('LOADING PRETRAINED MODEL..')
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']

        # reshape positional embedding to accomodate for image resolution change
        pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
        state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped

        if not args.evaluate:
            if config['distill']:
                m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                             model.visual_encoder_m)
                state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped

        for key in list(state_dict.keys()): #to match the key name when the pretrained model used BertForMaskedLM(which contains BertModel), but fine-tuning model used BertModel
            if 'bert' in key:
                new_key = key.replace('bert.', '')
                state_dict[new_key] = state_dict[key]
                del state_dict[key]

        msg = model.load_state_dict(state_dict, strict=False)
        print('load checkpoint from %s' % args.checkpoint)
        print(msg)

    model = model.to(device)
    model_without_ddp = model

    arg_opt = config['optimizer']
    optimizer = optim.AdamW(model.parameters(), lr=arg_opt['lr'], weight_decay=arg_opt['weight_decay'])

    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer)

    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']
    best = 0
    best_epoch = 0

    start_time = time.time()

    for epoch in range(0, max_epoch):
        if not args.evaluate:
            print('TRAIN')
            train_stats = train(model, train_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler,
                                config)

        print('VALIDATION')
        val_stats = evaluate(model, val_loader, tokenizer, device)
        print('TEST')
        test_stats = evaluate(model, test_loader, tokenizer, device)

        if not args.evaluate:
            if val_stats > best:
                save_obj = {
                    'model': model_without_ddp.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'lr_scheduler': lr_scheduler.state_dict(),
                    'config': config,
                    'epoch': epoch,
                }
                torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_best.pth'))
                best = val_stats
                best_epoch = epoch
        if args.evaluate:   break
        lr_scheduler.step(epoch + warmup_steps + 1)

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--output_dir', default='./output/VE')
    #parser.add_argument('--checkpoint', default='./output/VE/checkpoint_best.pth')
    parser.add_argument('--checkpoint', default='./Pretrain/checkpoint_00.pth')
    parser.add_argument('--text_encoder', default='bert-base-uncased')
    parser.add_argument('--evaluate', default=False)    #"True" for validation&test only
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training')
    parser.add_argument('--distributed', default=False, type=bool)
    args = parser.parse_args()

    cls_config = {
        'image_res': 256,
        'batch_size_train': 4,
        'batch_size_test': 4,
        'alpha': 0.4,
        'distill':True,
        'warm_up':False,
        'bert_config': './config_bert.json',
        'schedular': {'sched': 'cosine', 'lr': 2e-5, 'epochs': 5, 'min_lr': 1e-6,
                      'decay_rate': 1, 'warmup_lr': 1e-5, 'warmup_epochs': 1, 'cooldown_epochs': 0},
        'optimizer': {'opt': 'adamW', 'lr': 2e-5, 'weight_decay': 0.02}
    }

    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args, cls_config)