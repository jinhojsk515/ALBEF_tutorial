# ALBEF_tutorial
unofficial ALBEF tutorial, with MIMIC-CXR data.

## Model description
The following image from the ALBEF paper(https://arxiv.org/abs/2107.07651) illustrates the overall model architecture and training objectives of ALBEF. You can find more description of the detailed code in the comments in `pretrain_albef.py`.
![image](https://user-images.githubusercontent.com/59189526/210308191-059ace87-e4d0-4ee5-b743-68fb49c0a271.png)



## Data preparation
Data are not included in this repository. You need to prepare jsonl file(contains labels, etc.) and images. 

jsonl file: Visit https://github.com/SuperSupermoon/MedViLL, and from their `/data/mimic`, take `Train.jsonl, Valid.jsonl, Test.jsonl` and move them into your depository's `./data/MIMIC_CXR/`. This jsonl file contains the image name(as a path), english prescription, and diagnosis labels.

images: Visit https://github.com/SuperSupermoon/MedViLL, and from their README.md, download MIMIC-CXR datasets to get `mimic_dset.tar.gz`. Move this into your depository and unzip it. This will create `./re_512_3ch/` folder with photos.

> ※The image download link above is now blocked, since redistributing the MIMIC-CXR data is forbidden. Please visit their website https://physionet.org/content/mimic-cxr/2.0.0/ and follow the official method to access the data.

## Code running
Arguments can be passed with commands, or edited manually in the running code. Default values are already good to go, but I recommend you to check the arguments and hyperparameters in the code. You can read the code in `pretrain_albef.py` to understand how ALBEF works.

-Pretrain

```
python train.py
```

-Downstream task: multi-label classification

```
python classification.py --checkpoint ./save/file/path
```

-Retrieval [*pre-trained ALBEF would already work well for retrieval of its pre-train data, so no additional training.]

```
python retrieval.py --checkpoint ./save/file/path
```
