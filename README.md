#  Edge-Guided Multi-Scale Frequency Attention Network for Revealing the Spatial Distribution of Immune Cells in Tumor Microenvironment


##  Requirements

* torch
* torchvision 
* tqdm
* opencv
* scipy
* skimage
* PIL
* numpy

##  Usage

####  1. Training

```bash
python train.py  --root /path-to-project  --mode train
--train_data_dir /path-to-train_data   --valid_data_dir  /path-to-valid_data
```



####  2. Inference

```bash
python test.py  --root /path-to-project  --mode test  --load_ckpt checkpoint  
--test_data_dir  /path-to-test_data
```






