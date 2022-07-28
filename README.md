# CDP

This repository is for the paper "Bridge the Gap Between Architecture Spaces via A Cross-Domain Predictor".



## Preparation

### Dependencies

This code is tested on Python 3.6.9, and the dependent packages are listed:

- nasbench (see https://github.com/google-research/nasbench)
- nas_201_api (see https://github.com/D-X-Y/NAS-Bench-201)
- tensorflow (==1.15.0)
- pytorch
- matplotlib
- scipy
- ptflops

### Dataset

Two datasets are required:

- NAS-Bench-101
- NAS-Bench-201

Specifically, you can download these two datasets from https://storage.googleapis.com/nasbench/nasbench_only108.tfrecord and https://drive.google.com/open?id=16Y0UwGisiouVRxW-W5hEtbxmcHw_0hF_, separately. And then put these two datasets under the folder *path*.

The licenses for the datasets can be found in their respective github repositories.



## How to use

#### Search for architectures in DARTS with CDP

```
python cross_domain_predictor.py
```



#### Train the searched architecture on CIFAR-10

```
python train_cifar10.py --data your_dataset_path
```



#### Train the searched architecture on ImageNet

```
python train_imagenet.py --tmp_data_dir your_dataset_path
```



#### Ablation study on dataset tiny DARTS

```
python read_darts_dataset.py
```



In addition, you can adjust the parameters following the helps in these files.
