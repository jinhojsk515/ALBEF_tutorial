# ALBEF_tutorial
ALBEF tutorial, with MIMIC-CXR data

## Data preparation
You need to prepare jsonl file(contains labels, etc.) and images. 

jsonl file: Visit https://github.com/SuperSupermoon/MedViLL, and from their `/data/mimic`, take `Train.jsonl, Valid.jsonl, Test.jsonl` and move them into your depository's `./data/MIMIC_CXR/`.

images: Visit https://github.com/SuperSupermoon/MedViLL, and from their README.md, download MIMIC-CXR datasets to get `mimic_dset.tar.gz`. Move this into your depository and unzip it. This will create `./re_512_3ch/` folder with photos.

## Code running
Arguments can be passed with commands, or added manually in the code. Default values are already enough to run, but please check them in the code.

-Pretrain

```
python train.py
```

-Downstream task: multi-label classification

```
python classification.py
```
