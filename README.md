# Contents
Source code for the paper "SigDA: A Superimposed Domain Adaptation Framework for Automatic Modulation Classification", which is accepted by IEEE Transactions on Wireless Communications.

- [Contents](#Contents)
- [Abstract](#Abstract)
- [Dataset](#dataset)
- [Environment Requirements](#environment-requirements)
- [Script Description](#script-description)
    - [Script and Sample Code](#script-and-sample-code)
        - [Script Parameters](#script-parameters)
        - [Training Process](#training-process)         
        - [Evaluation Process](#evaluation-process)   
        - [M2SFE Series](#M2SFE-Series)
        - [Citation](#Citation)





# [Abstract](#Contents)
Due to the uncertainty of non-cooperative communication channels, the received signals often contain various impairment factors, leading to a significant decline in the performance
of existing deep learning (DL)-based automatic modulation
classification (AMC) models. Several preliminary works utilize
domain adaptation (DA) to alleviate this issue, however, they
are constrained by singular domain difference factor, whereas in
practice, these factors often manifest cumulatively. Therefore, this
paper introduce a more realistic task named superimposed DA,
where multiple domain difference factors are overlaid, reflecting
the cumulative nature of them. We propose the SigDA as a
solution framework, which adopts adversarial training to align
the data distribution in different domains. Two technical modules,
Multi-task based Masked Signal Feature Extractor (M2SFE) and
Signal Feature Pyramid Aggregation (SFPA), are innovatively
designed in SigDA. M2SFE utilizes mask and reconstruction
task to enhance feature extraction and achieves discriminative
feature selection through the design of feature mapping layers,
while SFPA can solve the problem of inconsistent signal length
in superimposed DA and can aggregate the features of signals
into the same dimension. We consider and superimpose various
typical signal domain difference factors, comprehensive experiments demonstrate that the proposed framework can achieve
significant performance improvement in various communication
channels.


# [Dataset](#Contents)

For publicly available datasets, we use RML2016.10a 
and RML2016.04c,you can download the dataset [here](https://www.deepsig.ai/datasets/).

For the generated typical dataset, we have considered AWGN, Rician, and Rayleigh channels, and will open source the dataset to the current repository in the future.

Please download and unzip the dataset and place it in the `.Datasets` folder. After successful extraction, the `.Datasets` folder should contain the following files: 
2016.04C.multisnr.pkl,RML2016.10a_dict.pkl.

Both of them contain data of 11
modulation types (BPSK, QPSK, 8PSK, 16QAM, 64QAM,
BFSK, CPFSK, PAM4, WB-FM, AM-SSB, AM-DSB) under
20 different SNRs range from -20dB to 18dB. Each sample
contains a complex number of 128 points. As a more standard
dataset, for each SNR in each category of modulated signals,
RML2016.10a contains 1000 samples, which means that it
contains 220000 samples in total. The number of samples in
each category of RML2016.04c is not consistent, it contains
162060 samples in total.

# [Environment Requirements](#Contents)

These models are implemented in Pytorch, and the environment setting is:

* Python 3.7
* pytorch 1.7.0



# [Script Description](#Contents)

## [Script and Sample Code](#Contents)

  ```text

├── checkpoints
│   ├── a2c_backbone.pth
│   ├── a2c_discriminator.pth
│   └── m2sfe_10A.pth
├── comparative_experiment
│   ├── checkpoints                        #comparative experiment checkpoints
│   ├── cldnn                              #code for CLDNN model 
│   ├── cnn2                               #code for CNN2 model 
│   ├── daelstm                            #code for DAELSTM model 
│   ├── lstm                               #code for LSTM model 
│   ├── mcldnn                             #code for MCLDNN model 
│   ├── resnet                             #code for ResNet model 
│   └── vgg
├── Datasets                               #Dataset file 
│   ├── 2016.04C.multisnr.pkl              
│   └── RML2016.10a_dict.pkl
├── model
│   ├── config.py                          # Porcess config file
│   ├── domainclassifer.py                 # model for domainclassifer
│   ├── __init__.py                   
│   └── m2sfe.py                           # model for M2SFE
├── scripts
│   ├── run_eval_da.sh                     # shell script for domain adaptation evalation
│   ├── run_eval.sh                        # shell script for M2SFE evalation
│   ├── run_train_da.sh                    # shell script for domain adaptation training
│   └── run_train.sh                       # shell script for M2SFE eval
├── utils
│   ├── da_loader.py                       # process dataset, create dataloader for domain adaptation
│   ├── dataloader.py                      # process dataset, create dataloader for supervised training
│   └── utils.py                           # tools file
├── default_config.yaml                    # parameter configuration file
├── domain_adaptation.py                   # training Script for domain adaptation
├── train_m2sfe.py                         # training Script for supervised training of M2SFE
├── eval_domain_adaptation.py              # evalation Script for domain adaptation
├── eval_m2sfe.py                          # evalation Script for supervised training of M2SFE
└── README.md

  ```

## [Script Parameters](Contents)
Parameters for both training and evaluation can be set in config.py

- Config for SigDA

  ```python
  data_root: "Datasets"
  data_name_10A: "RML2016.10a_dict.pkl"
  data_name_04C: "2016.04C.multisnr.pkl"
  model_root: "checkpoints"                        # weight file
  checkpoints_10A: "m2sfe_10A.pth"                 # weight file for supervised training of RML2016.10A dataset
  checkpoints_04C: "m2sfe_04C.pth"                 # weight file for supervised training of RML2016.04C dataset
  checkpoints_DA: "a2c_backbone.pth"               # weight file for domain adaptation backbone 
  checkpoints_DA_disc: "a2c_discriminator.pth"     # weight file for domain adaptation discriminator 
  show_confusion_metrix: False

  Epoch_num: 300                                   # number of Epochs
  lr: 0.001                                        # learning rate of supervised training
  Batch_size: 64                                   # Training batch size
  da_lr_g: 0.00001                                 # learning rate of generator in domain adaptation process
  da_lr_d: 0.0001                                  # learning rate of discriminator domain adaptation porcess
  ```

For more configuration details, please refer the script `config.py`.

## [Training Process](#Contents)

### [Training](#Contents)

-train for M2SFE

```bash
export CUDA_VISIBLE_DEVICES=0
python train_m2sfe.py > train_m2sfe.log 2>&1 &

or
cd scripts
bash run_train.sh 
```
-train for Domain Adaptation
```bash
export CUDA_VISIBLE_DEVICES=0
python domain_adaptation.py > domain_adaptation.log 2>&1 &

or
cd scripts
bash run_train_da.sh 
```


## [Evaluation Process](#Contents)

### [Evaluation](#Contents)

- Evaluation for M2SFE

  ```bash
  python eval_m2sfe.py > eval_m2sfe.log 2>&1 &
  或
  bash run_eval.sh
  ```

  The above shell command will run in the background. You can view the results through the file `eval/eval_m2sfe.log`. 

- Evaluation for Domain Adaptation

  ```bash
  python eval_domain_adaptation.py > eval_domain_adaptation.log 2>&1 &
  或
  bash run_eval_da.sh
  ```
  The above shell command will run in the background. You can view the results through the file `eval/eval_domain_adaptation.log`.

## [M2SFE Series](#Contents)
- The current code is set for M2SFE large. If you want to train or test other models in the M2SFE series, such as M2SFE-tiny or M2SFE-medium, Please check the partial code of the reference model in `train_m2sfe.py`

```python
   from model.m2sfe import M2SFE as M2SFE
```
and modify it as follows:

```python
   from model.m2sfe import M2SFE_tiny as M2SFE    # M2SFE-tiny model
   from model.m2sfe import M2SFE_medium as M2SFE  # M2SFE-medium model
```

## [Citation](#Contents)
If you use this code for your research, please cite our [paper](https://ieeexplore.ieee.org/document/10557536):
```
@ARTICLE{10557536,
  author={Wang, Shuang and Xing, Hantong and Wang, Chenxu and Zhou, Huaji and Hou, Biao and Jiao, Licheng},
  journal={IEEE Transactions on Wireless Communications}, 
  title={SigDA: A Superimposed Domain Adaptation Framework for Automatic Modulation Classification}, 
  year={2024},
  volume={},
  number={},
  pages={1-1},
  keywords={Feature extraction;Task analysis;Modulation;Adaptation models;Convolution;Data models;Wireless communication;Automatic Modulation Classification;Domain Adaptation;Multi-task learning;Adversarial training},
  doi={10.1109/TWC.2024.3399067}}
```
