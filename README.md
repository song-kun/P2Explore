# P2Explore
Implementation of P^2 Explore: Efficient Exploration in Unknown Clustered Environment with Floor Plan Prediction


## News
- Feb. 26, 2025: P2 Explore is open-sourced.

## Overview
**P2 Explore** uses the predicted information for exploration in 2D environment. It mainly has two parts: 1. FPUNet to perform map prediction; 2. the implementation of using the predicted information for exploration.

This work is fully open-sourced. We open-sourced the code for training this NN and how to reproduce our result.

# Quick Start
## Platform
- No specific platform is required. You can use Linux and windows.

## Denpendency
### Denpendency for Python
We recommend you install the necessary package before running this code. We provide a *requirements.txt* file to show our environment. However, we do not recommend you install all the packages. You can run our code first, then install the necessary packages,

Some necessary packages are listed below.

- pytorch
- numpy, scikit-learn
- cv2 (if you have problem install this package, you can modify the way importing the map)

### Install Code for p2explore
```
mkdir ~/p2explore && cd ~/p2explore
git clone git@github.com:KunSong-L/P2Explore.git
cd P2Explore/dataset && unzip map.zip
```

## Download the checkpoint
```
cd ~/p2explore/P2Explore && mkdir trained_model
cd trained_model
```
You can download our checkpoint using [google drive](https://drive.google.com/file/d/1v0JAYG_m0ZG8ol_Em5d-EqevraClvgx3/view?usp=drive_link) or [Pan Baidu, 百度网盘](https://pan.baidu.com/s/17Gn4mTcmBvFBhF1zxpJbDg?pwd=xaad).

You should put this checkpoint into the folder *trained_model*.

## Train FPUNet
Firstly, you need to generate the dataset. You can run *1sim_explore.ipynb* to generate the dataset. This will take about 50 min.

Then, run *2train_fpunt.ipynb* to train the model. 

If you do not want to train this by yourself, you can directly use the checkpoint we provide above for exploration evaluation.

## Exploration Evaluation
You can run *3Pre_explore.ipynb* for evaluation.

If you what to visulize the topology connectio between rooms during exploration, you can turn the `vis_flag` to True.

To generate *node_dis_file*, you can change **dir_use**
in
```
now_agent = agent(now_sim.agent_map,now_sim.pose,now_sim.map,node_path=node_path,dir_use=True)
```
to False in the first time.

Except the first time, you can turn **dir_use** into True, and this script will run faster.

# Code Overview
We will introduce our code briefly. Some important files or functions are listed below
- *FPUNet*: the stucture of FPUNet.
- `class agent`: the simulated robot that performs exploration.
- `class sim_env`: the simulation.

# Citation
If you use this code for your research, please cite our papers. *https://arxiv.org/abs/2409.10878*

```
@article{song2024p2,
  title={P2 Explore: Efficient Exploration in Unknown Clustered Environment with Floor Plan Prediction},
  author={Song, Kun and Chen, Gaoming and Tomizuka, Masayoshi and Zhan, Wei and Xiong, Zhenhua and Ding, Mingyu},
  journal={arXiv preprint arXiv:2409.10878},
  year={2024}
}
```
