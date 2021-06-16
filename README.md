# WiFi-based Cross-Domain Gesture Recognition via Modified Prototypical Networks

## About the code files

### models (folder)
These are the different types of Dual-PAth PN
* LSTM_CSI.py: the network is based on CNN and LSTM (Model_type_1).
* MobiV3_CSI_model.py: the network is based on the 2D convolutional network (Model_type_2).
* ResNet_CSI_model.py: the network is based on the 1D convolutional network (Model_type_3) (Using in WiGr).

* Prototypical_CnnLstmNet.py: the pytorch-lightning version of Model_type_1;
* Prototypical_2DMobileNet.py: the pytorch-lightning version of Model_type_2;
* Prototypical_1DResNet.py: the pytorch-lightning version of Model_type_3 (Using in WiGr).

### reimplement (folder)
Reimplementation of the related models: Widar3.0, EI, JADA, SignFi, ARIL, WiAG

## Training a Dual-Path Prototypical Network

### Install dependencies

* This code has been tested on Ubuntu 16.04 with Python 3.6 and PyTorch-1.8.0.
* Install [PyTorch and torchvision](http://pytorch.org/).
* Install (pytorch-lightning)[https://github.com/PyTorchLightning/pytorch-lightning]

### Download the datasets

* Download Widar3.0 dataset: http://tns.thss.tsinghua.edu.cn/widar3.0/
* Download ARIL dataset:
* Download CSIDA dataset:


### Train and Test the model

* Run `python in_domain_run.py`. This will run in-domain training and place the results into `lightning_logs` (this folder will be automatic constructed).
* Run `python ross_domain_run.py`. This will run cross-domain training
* the parameter_config.py is the configurations of the cross-domain and the in-domain experiments

