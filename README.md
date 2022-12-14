# ALBEF_tutorial
ALBEF tutorial, with MIMIC-CXR data.

!! retrieval.py is not verified right now. It will be fixed soon. !!


## Data preparation
Data are not included in this repository. You need to prepare jsonl file(contains labels, etc.) and images. 

jsonl file: Visit https://github.com/SuperSupermoon/MedViLL, and from their `/data/mimic`, take `Train.jsonl, Valid.jsonl, Test.jsonl` and move them into your depository's `./data/MIMIC_CXR/`. This jsonl file contains the image name(as a path), english prescription, and diagnosis labels.

images: Visit https://github.com/SuperSupermoon/MedViLL, and from their README.md, download MIMIC-CXR datasets to get `mimic_dset.tar.gz`. Move this into your depository and unzip it. This will create `./re_512_3ch/` folder with photos.

## Code running
Arguments can be passed with commands, or edited manually in the running code. Default values are already good to go, but I recommend you to check the arguments and hyperparameters in the code. You can read the code in `pretrain_albef.py` to understand how ALBEF works.

-Pretrain

```
python train.py
```

-Downstream task: multi-label classification

```
python classification.py
```

-Retrieval [*pre-trained ALBEF would already work well for retrieval of its pre-train data, so no additional training.]

```
python retrieval.py
```
