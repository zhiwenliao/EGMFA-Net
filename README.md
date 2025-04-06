Requirements\n
torch
torchvision
tqdm
opencv
scipy
skimage
PIL
numpy
Usage
1. Training
python train.py  --root /path-to-project  --mode train
--train_data_dir /path-to-train_data   --valid_data_dir  /path-to-valid_data
2. Inference
python test.py  --root /path-to-project  --mode test  --load_ckpt checkpoint  
--test_data_dir  /path-to-test_data
