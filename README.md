# ST-CDPRN
ST-CDPRN:Self-Training Conditional Diffusion for Robust 3D Point Cloud Reconstruction
# Dependencies
refer requirements.txt for the required packages.
# Download data
imagefile  dataset/ShapeNetRendering
voxelfile  dataset/ShapeNetVox32
# Train
| Package      | Version                          |
| ------------ | -------------------------------- |
| PyTorch      | ≥ 1.6.0                          |
| h5py         | *not specified* (we used 4.61.1) |
| tqdm         | *not specified*                  |
| tensorboard  | *not specified* (we used 2.5.0)  |
| numpy        | *not specified* (we used 1.20.2) |
| scipy        | *not specified* (we used 1.6.2)  |
| scikit-learn | *not specified* (we used 0.24.2) |
diffusion
```bash
# Train an auto-encoder
python train_ae.py 

# Train a generator
python train_gen.py
```
SSP3D
train ShapeNet stage1
```
python runner_shapenet.py
```
train ShapeNet stage2
```
python runner_shapenet.py --finetune --weights=xxx.pth
```

test ShapeNet
```
python runner_shapenet.py --test --weights=xxx.pth
```
# our result
