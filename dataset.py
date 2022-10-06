from torch.utils.data import Dataset
import pandas
import torch
from PIL import Image
from torchvision.transforms import transforms
import os
import json
import jsonlines
from utils import pre_caption



class MIMIC_CXRDataset(Dataset):
    def __init__(self, data_path, transform=None, data_length=None):
        self.transform = transform
        self.data_dir = os.path.dirname(data_path)
        self.data = [json.loads(l) for l in open(data_path)]
        if data_length is not None: self.data=self.data[:data_length]
        self.ltoi = ['','Pneumothorax', 'Pleural Other', 'No Finding', 'Cardiomegaly', 'Atelectasis',
                     'Consolidation', 'Support Devices', 'Edema', 'Pneumonia', 'Pleural Effusion',
                     'Enlarged Cardiomediastinum', 'Fracture', 'Lung Opacity', 'Lung Lesion']



    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        info=self.data[index]
        img_path=info['img'].replace('/home/mimic-cxr/dataset/image_preprocessing','.')
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)

        raw_labels=[w.strip().replace('\'', '') for w in info['label'].split(',')]
        labels=torch.zeros(len(self.ltoi))
        for l in raw_labels:    labels[self.ltoi.index(l)]=1

        return img, info['text'], labels