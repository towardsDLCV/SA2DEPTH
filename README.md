<div align="center">
  
<h1>SA2DEPTH</h1>
    
</div>

## Installation
- Creating a conda virtual environment and install packages
```bash
conda create -n SA2DEPTH python=3.9
conda activate SA2DEPTH
pip install torch==2.1.0 torchvision==0.16.0 --index-url https://download.pytorch.org/whl/cu118
pip install matplotlib, tqdm, tensorboardX, timm, mmcv, open3d, einops
# SSM package
pip install causal_conv1d==1.1.0
pip install mamba_ssm==1.1.1
```

- Install DCNv3 for backbone
```bash
pip install -U openmim
mim install mmcv-full==1.5.0
pip install timm==0.6.11 mmdet==2.28.1
pip install opencv-python termcolor yacs pyyaml scipy
```

Then, compile the CUDA operators by executing the following commands:
```bash
cd ./sa2depth/ops_dcnv3
sh ./make.sh
python test.py
# All checks must be True, and the time cost should be displayed at the end.
```

Also, you can install DCNv3 according to [InternImage](https://github.com/OpenGVLab/InternImage/tree/master).

## Datasets
You can prepare the datasets KITTI and NYUv2 according to [here](https://github.com/cleinc/bts/tree/master/pytorch) and download the SUN RGB-D dataset from [here](https://rgbd.cs.princeton.edu/), and then modify the data path in the config files to your dataset locations.


## Training
Training the NYUv2 model:
```
python sa2depth/train.py configs/arguments_train_nyu.txt
```

Training the KITTI_Eigen model:
```
python sa2depth/train.py configs/arguments_train_kittieigen.txt
```

## Evaluation
Evaluate the NYUv2 model:
```
python sa2depth/eval.py configs/arguments_eval_nyu.txt
```

Evaluate the NYUv2 model on the SUN RGB-D dataset:
```
python sa2depth/eval_sun.py configs/arguments_eval_sun.txt
```

Evaluate the KITTI_Eigen model:
```
python sa2depth/eval.py configs/arguments_eval_kittieigen.txt
```

