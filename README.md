<div align="center">
  
<h1>SA<sub>2</sub>Depth: Toward Smooth Depth Driven by Selective Attention and Selective Aggregation</h1>

<div>
    <a href='https://scholar.google.com/citations?user=5C9TeqgAAAAJ&hl=ko&oi=sra' target='_blank'>Cheolhoon Park</a><sup>1</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=4Q-TY8YAAAAJ&hl=ko' target='_blank'>Woojin Ahn</a><sup>2</sup>&emsp;
    <a href='https://scholar.google.com/citations?user=SIfp2fUAAAAJ&hl=ko&oi=sra' target='_blank'>Hyunduck Choi</a><sup>3,*</sup>&emsp;
</div>
<div>
    <sup>1</sup>Korea University, <sup>2</sup>Inha University, <sup>3</sup>SeoulTech
</div>


<div>
    <h4 align="center">
        • <a href="https://ieeexplore.ieee.org/xpl/RecentIssue.jsp?punumber=6046" target='_blank'>IEEE TMM 2025</a> •
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

## Models
| Model | Abs Rel | Sq Rel | RMSE | a1 | a2 | a3| Link|
| ------------ | :---: | :---: | :---: |  :---: |  :---: |  :---: |  :---: |
|NYUv2 (S)| 0.080 | 0.035 | 0.292 | 0.945 | 0.994 | 0.999 |[[Google]](https://drive.google.com/file/d/1LQCCZ9i9jtcZIF5O5aAbBexWrpal9h9i/view?usp=drive_link)|
|NYUv2 (T)| 0.096 | 0.050 | 0.340 | 0.915 | 0.988 | 0.997 |[[Google]](https://drive.google.com/file/d/1_08VRPD1dcn7x0Oi4HNEnEu2m3jUaQbR/view?usp=drive_link)|
|KITTI_Eigen (S)| 0.052 | 0.159 | 2.158 | 0.973 | 0.997 | 0.999 |[[Google]](https://drive.google.com/file/d/1zk-1cJty6kojMBEFmlOtbovJ4SED71cj/view?usp=drive_link)|
|KITTI_Eigen (L)| 0.048 | 0.137 | 2.026 | 0.979 | 0.998 | 0.999 |[[Google]](https://drive.google.com/file/d/1UF4YZEmGs0Kxoyck3IsAm5KddbDE4UUF/view?usp=drive_link)|

| Model | SILog | Abs Rel | Sq Rel | iRMSE |
| ------------ | :---: | :---: | :---: | :---: |
|KITTI_Official| 9.63 | 7.91 | 1.64 | 10.33|

| Model | Abs Rel | RMSE | a1 | a2 |
| ------------ | :---: | :---: | :---: | :---: |
|DIML-CVD | 0.162 | 4.149 | 0.810 | 0.939 |

## Acknowledgement

Our code is based on the implementation of [NeWCRFs](https://github.com/aliyun/NeWCRFs), [BTS](https://github.com/cleinc/bts) and [IEBins](https://github.com/ShuweiShao/IEBins/tree/main). We thank their excellent works.
