# ARN-LSTM-final
## Code for the paper "ARN-LSTM:A Multi-Stream Attention-Based Model for Action Recognition with Temporal Dynamics".
It contains an implementation of our Attention Relation Long Shor-term Memory Network (ARN-LSTM ), an end-to-end NN model for Group Activity Recognition using Skeleton information. 


<div align="center">
    <img src="./images/cspc2024.jpg", width="900",height='600'>
</div>

CSPC 2024 Website: Visit the CSPC website for details https://cspc.cs.usm.my/home

# The code details will be soon when it is sorted well and the paper be published.

## 1 Paper Details
**ARN-LSTM:A Multi-Stream Attention-Based Model for Action Recognition with Temporal Dynamics**[link](xxxxx)
>Chuanchaun Wang, Ahmad Sufril Azlan Mohmamed, Xiao Yang, Xiang Li

## 2 Prerequisites
## Requirements

  ![Python >=3.8.16](https://img.shields.io/badge/Python->=3.8.16-yellow.svg)    ![Tensorflow >=2.8.0](https://img.shields.io/badge/Tensorflow->=2.8.0-blue.svg)

## Installation

- Python: 3.8.16
- TensorFlow: 2.8.0
- misc: argparse, numpy, pandas. etc


### 2.2 Experimental Dataset
## Dataset setup
For all the datasets, be sure to read and follow their license agreements, and cite them accordingly. The datasets we used are as follows:
- [NTU RGB+D 60](https://arxiv.org/pdf/1604.02808.pdf)
- [NTU RGB+D 120](https://arxiv.org/pdf/1905.04757.pdf)


## Data Availability
The datasets used or analyzed during the current study are available from the corresponding author upon reasonable request. The raw dataset was downloaded from the dataset home page https://rose1.ntu.edu.sg/dataset/actionRecognition/.

There are 302 samples of **NTU RGB+D 60** and 532 samples of **NTU RGB+D 120** need to be ignored, which are shown in the project **'src/dataset/NTU.py'** [link](https://github.com/shahroudy/NTURGB-D/blob/master/Matlab/NTU_RGBD_samples_with_missing_skeletons.txt).


## 3 Parameters


## 4 Running


## 5 Results and Pretrained Models
Results from our proposed method in the NTU RGB+D 120 dataset,with Cross-view benchmark:

Method | Accuracy
------------ | -------------
ARN-LSTM(joint+motion)					| 91.8%
ARN-LSTM(temporal+motion)			| 94.3%
ARN-LSTM(joint+temporal+motion)				| 95.8%

Results from our proposed method in the NTU RGB+D 120 dataset,with Cross-subject benchmark:

Method | Accuracy
------------ | -------------
ARN-LSTM(joint+motion)					| 94.9%
ARN-LSTM(temporal+motion)			| 97.7%
ARN-LSTM(joint+temporal+motion)				| 99.7%


## 6 Citation



## Acknowledgement
We are very grateful for these excellent work [ST-GCN](https://github.com/yysijie/st-gcn),[IRN](https://github.com/mauriciolp/inter-rel-net),[PSTL](https://github.com/YujieOuO/PSTL), some thinks borrow from them.

## Licence
This project is licensed under the terms of the MIT license.

