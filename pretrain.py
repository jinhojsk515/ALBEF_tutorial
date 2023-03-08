import argparse
from tqdm import tqdm
from pathlib import Path
import utils
import torch
import numpy as np
import random
from pretrain_albef import ALBEF
import time
import os
from vit import interpolate_pos_embed
import torch.backends.cudnn as cudnn
from transformers import BertTokenizer
import torch.distributed as dist
import datetime
from torchvision import transforms
from PIL import Image
import torchvision.transforms as T
from dataset import MIMIC_CXRDataset
from torch.utils.data import DataLoader
import torch.optim as optim
from scheduler import create_scheduler


def train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, scheduler, config):
    # train
    model.train()

    header = 'Train Epoch: [{}] '.format(epoch)
    print_freq = 50
    step_size = 100
    warmup_iterations = warmup_steps * step_size

    if args.distributed:
        data_loader.sampler.set_epoch(epoch)

    tqdm_data_loader=tqdm(data_loader, miniters=print_freq, desc=header)
    for i, (image, text, _) in enumerate(tqdm_data_loader):
        optimizer.zero_grad()
        image = image.to(device, non_blocking=True)
        text_input = tokenizer(text, padding='longest', truncation=True, max_length=250, return_tensors="pt").to(device)

        if epoch > 0:
            alpha = config['alpha']
        else:
            alpha = config['alpha'] * min(1., i / len(data_loader))

        loss_mlm, loss_ita, loss_itm = model(image, text_input, alpha=alpha)
        loss = loss_mlm + loss_ita + loss_itm
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.)
        optimizer.step()

        tqdm_data_loader.set_description(f'loss_mlm={loss_mlm.item():.4f}, loss_ita={loss_ita.item():.4f}, loss_itm={loss_itm.item():.4f}, lr={optimizer.param_groups[0]["lr"]:.6f}')

        if epoch == 0 and i % step_size == 0 and i <= warmup_iterations:
            scheduler.step(i // step_size)



def main(args,config):
    print('aa',args.distributed,config['image_res'])
    utils.init_distributed_mode(args)

    device = torch.device(args.device)

    # fix the seed for reproducibility
    seed = args.seed + utils.get_rank()
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    cudnn.benchmark = True

    start_epoch = 0
    max_epoch = config['schedular']['epochs']
    warmup_steps = config['schedular']['warmup_epochs']

    #### Dataset ####
    normalize = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    pretrain_transform = transforms.Compose([
        transforms.RandomResizedCrop(config['image_res'], scale=(0.5, 1.0), interpolation=T.InterpolationMode.BICUBIC),
        #RandomAugment(2, 7, isPIL=True, augs=['Identity', 'AutoContrast', 'Equalize', 'Brightness', 'Sharpness',
        #                                      'ShearX', 'ShearY', 'TranslateX', 'TranslateY', 'Rotate']),
        transforms.ToTensor(),
        normalize,
    ])
    print("Creating dataset")
    dataset = MIMIC_CXRDataset("./data/MIMIC_CXR/Train.jsonl",transform=pretrain_transform)
    if args.distributed:
        num_tasks = utils.get_world_size()
        global_rank = utils.get_rank()
        #samplers = create_sampler(datasets, [True], num_tasks, global_rank)
        samplers=[None]
    else:
        samplers = [None]

    data_loader=DataLoader(dataset,batch_size=config['batch_size'],pin_memory=True,drop_last=True)
    tokenizer = BertTokenizer.from_pretrained(args.text_encoder)

    #### Model ####
    print("Creating model")
    model = ALBEF(config=config, text_encoder_name=args.text_encoder, tokenizer=tokenizer, init_vit=True)
    model = model.to(device)
    print('#parameters:',sum(p.numel() for p in model.parameters() if p.requires_grad))

    arg_opt = config['optimizer']
    optimizer=optim.AdamW(model.parameters(),lr=arg_opt['lr'],weight_decay=arg_opt['weight_decay'])

    arg_sche = utils.AttrDict(config['schedular'])
    lr_scheduler, _ = create_scheduler(arg_sche, optimizer) #You can just use a given scheduler from pytorch

    #loading if there's a checkpoint
    if args.checkpoint:
        checkpoint = torch.load(args.checkpoint, map_location='cpu')
        state_dict = checkpoint['model']
        if args.resume:
            optimizer.load_state_dict(checkpoint['optimizer'])
            lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
            start_epoch = checkpoint['epoch'] + 1
        else:
            pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder.pos_embed'], model.visual_encoder)
            m_pos_embed_reshaped = interpolate_pos_embed(state_dict['visual_encoder_m.pos_embed'],
                                                         model.visual_encoder_m)
            state_dict['visual_encoder.pos_embed'] = pos_embed_reshaped
            state_dict['visual_encoder_m.pos_embed'] = m_pos_embed_reshaped
        model.load_state_dict(state_dict)
        print('load checkpoint from %s' % args.checkpoint)

    model_without_ddp = model
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module

    print("Start training")
    start_time = time.time()

    for epoch in range(start_epoch, max_epoch):

        if epoch > 0:
            lr_scheduler.step(epoch + warmup_steps)

        if args.checkpoint:
            stopped_epoch=int(args.checkpoint[-6:-4])
            if epoch<=stopped_epoch:    continue


        train_stats = train(model, data_loader, optimizer, tokenizer, epoch, warmup_steps, device, lr_scheduler, config)
        #if utils.is_main_process():
        if True:
            print('SAVE START')
            save_obj = {
                'model': model_without_ddp.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_scheduler': lr_scheduler.state_dict(),
                'config': config,
                'epoch': epoch,
            }
            torch.save(save_obj, os.path.join(args.output_dir, 'checkpoint_%02d.pth' % epoch))
            print('SAVE DONE','checkpoint_%02d.pth' % epoch)

        #dist.barrier()

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--checkpoint', default='') #checkpoint name if you want to start from the paused model
    parser.add_argument('--resume', default=False, type=bool)
    parser.add_argument('--output_dir', default='./Pretrain/')
    parser.add_argument('--text_encoder', default='bert-base-uncased')  #name of pretrained BERT model and tokenizer
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--world_size', default=1, type=int, help='number of distributed processes')    #for multi-gpu
    parser.add_argument('--dist_url', default='env://', help='url used to set up distributed training') #for multi-gpu
    parser.add_argument('--distributed', default=False, type=bool)  #for multi-gpu
    args = parser.parse_args()

    pretrain_config = {
        'image_res': 256,#256
        'vision_width': 768,#768
        'embed_dim': 256,#256
        'batch_size': 4,#64
        'temp': 0.07,
        'mlm_probability': 0.15,
        'queue_size': 2048,#65536
        'momentum': 0.995,
        'alpha': 0.4,
        'bert_config': './config_bert.json',    #config file for BERT model. The configuration for ViT can be manually changed in albef.py
        'schedular': {'sched': 'cosine', 'lr': 1e-4, 'epochs': 30, 'min_lr': 1e-5,
                      'decay_rate': 1, 'warmup_lr': 1e-5, 'warmup_epochs': 20, 'cooldown_epochs': 0},
        'optimizer': {'opt': 'adamW', 'lr': 1e-4, 'weight_decay': 0.02}
    }
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    main(args, pretrain_config)
