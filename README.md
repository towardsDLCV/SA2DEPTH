<div align="center">
  
<h1>SA<sub>2</sub>Depth: Toward Smooth Depth Driven by Selective Attention and Selective Aggregation</h1>

<div>
    <a href='https://scholar.google.com/citations?user=5C9TeqgAAAAJ&hl=ko&oi=sra' target='_blank'>Cheolhoon Park</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=4Q-TY8YAAAAJ&hl=ko' target='_blank'>Woojin Ahn</a><sup>2</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=SIfp2fUAAAAJ&hl=ko&oi=sra' target='_blank'>Hyunduck Choi</a><sup>3,*</sup>&emsp;
</div>
<div>
    <sup>1</sup>Chonnam National University, <sup>2</sup>Korea University, <sup>3</sup>SeoulTech
</div>


<div>
    <h4 align="center">
        • <a href="" target='_blank'>Arxiv 2024</a> •
    </h4>
</div>

## Abstract

<div style="text-align:center">
</div>

</div>

>The challenges in single-image depth prediction (SIDP) are mainly due to the lack of smooth depth ground truth and the presence of irregular and complex objects. While window-based attention mechanisms, which balance long-range dependency capture with computational efficiency by processing elements within a fixed grid, have advanced SIDP research, they are limited by a constrained search range. This limitation can impede smooth depth estimation in irregularity and complexity. To address these challenges, we propose a novel attention mechanism that selectively identifies and aggregates only the most relevant information. Our approach enables flexible and efficient exploration by using data-dependent movable offsets to select substantial tokens and designating them as key-value pairs. Furthermore, we overcome the issue of small softmax values in traditional attention mechanisms through score-based grouping with top-k selection. Our feed-forward network, which incorporates a gating mechanism and grouped convolutions with varying cardinalities, refines features before passing them to subsequent layers, allowing for targeted focus on input features. Finally, we utilize feature maps from hierarchical decoders to estimate bin centers and per-pixel probability distributions. We introduce a 4-way selective scanning technique to aggregate these perpixel probability distributions smoothly, resulting in a dense and continuous depth map. The proposed network, named selective attention and selective aggregate depth (SA<sub>2</sub>Depth), demonstrates state-of-the-art performance across multiple datasets compared to previous methods.

</div>

## Installation
- Creating a conda virtual environment and install packages
```bash
conda create -n SADE python=3.9
conda activate SADE
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
cd ./sade/ops_dcnv3
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
python sade/train.py configs/arguments_train_nyu.txt
```

Training the KITTI_Eigen model:
```
python sade/train.py configs/arguments_train_kittieigen.txt
```

## Evaluation
Evaluate the NYUv2 model:
```
python sade/eval.py configs/arguments_eval_nyu.txt
```

Evaluate the NYUv2 model on the SUN RGB-D dataset:
```
python sade/eval_sun.py configs/arguments_eval_sun.txt
```

Evaluate the KITTI_Eigen model:
```
python sade/eval.py configs/arguments_eval_kittieigen.txt
```

