# 8501-Project

## Introduction

Our work is centered around an existing tool, MMPose. MMPose is an open-source toolbox for pose estimation based on PyTorch. It is a part of the OpenMMLab project. 

What I want to state here is that although we use the code in MMPose, we do not simply copy it. On the contrary, we fully understand the complete code running logic of each part of the code. Moreover, what we use are very mature codes that are already open source. Because these codes are already very mature, we think there is no need to rewrite them. On the contrary, we have fully commented the core code parts we use.

We just use MMPose as a tool to help us save time to quickly train the model and decide on subsequent adjustments based on the results. It is worth noting that the parameters of the model are still defined by ourselves, and we provide the Config script for each model we run as well as the task script submitted to ANU DUG.

At the same time, we also spent some initial time configuring and testing the mmpose environment on the ANU DUG. This was not a simple thing, so I think this can also be counted as the workload of our team. We provide complete instructions for configuring MMPose on ANU DUG.

In the early days of the project, we spent some money configuring and testing the mmpose environment on the ANU DUG. This was not a very easy task, so I think this can also be counted as one of the team's workload. At the same time, we provide a complete instruction manual for configuring MMPose on ANU DUG.

The code we mainly use includes three parts, the Backbone part, the Rleloss part and the NF part. The codes for these three parts are in corresponding folders. Among them, the backbone part includes the three backbones used in this article, namely alexnet, resnet and scnet. The Rleloss part includes RleHeadStructure and Rleloss used in this article. The NF part includes the NF model used in this article, RealNVP.
## About this directory
- `mmpose/mmpose/models/heads/regression_heads
/rle_head.py`:Top-down regression head introduced in `RLE`_ by Li et al(2021). The head is composed of fully-connected layers to predict the coordinates and sigma(the variance of the coordinates) together.
- `mmpose/mmpose/models/losses
/regression_loss.py`: Code about Residual Log-Likelihood Estimation Loss and SmoothL1Loss
- `mmpose/mmpose/models/utils
/realnvp.py`: Code about density estimation using Real NVP
- `mmpose/models/backbones/alexnet.py`: Code about AlexNet.
- `mmpose/mmpose/models/backbones
/resnet.py`: Code about ResNet backbon
- `mmpose/mmpose/models/backbones
/scnet.py`: Code about SCNet backbon
