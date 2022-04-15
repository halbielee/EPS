[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/railroad-is-not-a-train-saliency-as-pseudo-1/weakly-supervised-semantic-segmentation-on-4)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on-4?p=railroad-is-not-a-train-saliency-as-pseudo-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/railroad-is-not-a-train-saliency-as-pseudo-1/weakly-supervised-semantic-segmentation-on)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on?p=railroad-is-not-a-train-saliency-as-pseudo-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/railroad-is-not-a-train-saliency-as-pseudo-1/weakly-supervised-semantic-segmentation-on-1)](https://paperswithcode.com/sota/weakly-supervised-semantic-segmentation-on-1?p=railroad-is-not-a-train-saliency-as-pseudo-1)

## Railroad is not a Train: Saliency as Pseudo-pxiel Supervision for Weakly Supervised Semantic Segmentation (CVPR 2021)

[CVPR 2021 paper](https://openaccess.thecvf.com/content/CVPR2021/html/Lee_Railroad_Is_Not_a_Train_Saliency_As_Pseudo-Pixel_Supervision_for_CVPR_2021_paper.html)

Seungho Lee<sup>1,* </sup>, Minhyun Lee<sup>1,*</sup>, Jongwuk Lee<sup>2</sup>, Hyunjung Shim<sup>1</sup>

<sub>* indicates an equal contribution</sub>

<sup>1</sup> <sub>School of Integrated Technology, Yonsei University</sub>  
<sup>2</sup> <sub>Department of Computer Science of Engineering, Sungkyunkwan University</sub>  




## Introduction

![EPS](figure/figure_EPS.png)
Existing studies in weakly-supervised semantic segmentation (WSSS)
using image-level weak supervision have several limitations: 
sparse object coverage, inaccurate object boundaries, 
and co-occurring pixels from non-target objects. 
To overcome these challenges, we propose a novel framework, 
namely Explicit Pseudo-pixel Supervision (EPS), 
which learns from pixel-level feedback by combining two weak supervisions; 
the image-level label provides the object identity via the localization map 
and the saliency map from the off-the-shelf saliency detection model 
offers rich boundaries. We devise a joint training strategy to fully 
utilize the complementary relationship between both information. 
Our method can obtain accurate object boundaries and discard co-occurring pixels, 
thereby significantly improving the quality of pseudo-masks.


## Updates

12 Jul, 2021: Initial upload

19 Aug, 2021: Minor update on information about dCRF and the pre-trained model of the segmentation networks

- Please see the issuses: [dCRF](https://github.com/halbielee/EPS/issues/5) and [pre-trained model](https://github.com/halbielee/EPS/issues/4)

28 Aug, 2021: Major updates about MS-COCO 2014 dataset and minor updates (cleanup)

15 Apr, 2022: Minor update on information about the method setting up 'cls_labels.npy' the for ms-coco 17 dataset
- Please see the issue: [coco17](https://github.com/halbielee/EPS/issues/13)
## Installation


- Python 3.6
- Pytorch >= 1.0.0
- Torchvision >= 0.2.2
- MXNet
- Pillow
- opencv-python (opencv for Python)


## Execution



### Dataset & pretrained model
- PASCAL VOC 2012
    - [Images](http://host.robots.ox.ac.uk/pascal/VOC/voc2012/) 
    - [Saliency maps](https://drive.google.com/file/d/19AjSmgdMlIZH4FXVZ5zjlUZcoZZCkwrI/view?usp=sharing) 
      using [PFAN](https://arxiv.org/abs/1903.00179)

- MS-COCO 2014
    - [Images](https://cocodataset.org/#home) 
    - [Saliency maps](https://drive.google.com/file/d/1o50oztQqTc_xZdgpIEvgKD2Xi_HqBFig/view?usp=sharing) 
      using [PFAN](https://arxiv.org/abs/1903.00179)
    - [Segmentation masks](https://drive.google.com/file/d/16wuPinx0rdIP_PO0uYeCn9rfX2-evc-S/view?usp=sharing)

- Pretrained models
    - [ImageNet-pretrained Model](https://drive.google.com/file/d/15F13LEL5aO45JU-j45PYjzv5KW5bn_Pn/view?usp=sharing) 
      for [ResNet38](https://arxiv.org/abs/1611.10080)

- MS-COCO 2017
    - The way to setup "cls_labels.npy" for MS-COCO 2017 dataset. This work is done by [JimmyMa99](https://github.com/JimmyMa99/coco17-get-cls_label/tree/main) and please see [the codes](https://github.com/JimmyMa99/coco17-get-cls_label/tree/main).

### Classification network  
- Execute the bash file for training, inference and evaluation.
    ```bash
    # Please see these files for the detail of execution.
    
    # PASCAL VOC 2012 
    # Baseline
    bash script/vo12_cls.sh
    # EPS
    bash script/voc12_eps.sh
    
    # MS-COCO 2014
    # Baseline
    bash script/coco_cls.sh
    # EPS
    bash script/coco_eps.sh  
    ```
- We provide checkpoints, training logs, and performances for each method and each dataset.

  Please see the details from the script files.

  | Dataset         | METHOD | Train(mIoU) | Checkpoint                                                   | Training log                        |
  | --------------- | ------ | ----------- | ------------------------------------------------------------ | -------------------------------------- |
  | PASCAL VOC 2012 | Base   | 47.05       | [Download](https://drive.google.com/file/d/1dO4ZKerN6MMFLjaDw0TV_h7ZUOxRQvdq/view?usp=sharing)                                                 | [voc12_cls.log](log/log_voc12_cls.log) |
  | PASCAL VOC 2012 | EPS    | 69.22       | [Download](https://drive.google.com/file/d/1f3iVGRt2nH8BMxEP-w6VoJANouoHNLYM/view?usp=sharing)                                                 | [voc12_eps.log](log/log_voc12_eps.log) |
  | MS-COCO 2014    | Base   | 31.23       | [Download](https://drive.google.com/file/d/1VPi5GbTarzix_dwHBWC__1_WWrEZ6fzw/view?usp=sharing) | [coco_cls.log](log/log_coco_cls.log)   |
  | MS-COCO 2014    | EPS    | 37.15       | [Download](https://drive.google.com/file/d/1D9dDj2_oR_aLUWpex2HL3o80zbGpk7Vp/view?usp=sharing) | [coco_eps.log](log/log_coco_eps.log)   |


- dCRF hyper-parameters
  - We did not use dCRF for our pseudo-masks, but only used for the comparision in the paper.
  - We chose the hyper-parameters for dCRF used in ResNet101-based DeepLabV2 among other candidates([OAA](https://github.com/PengtaoJiang/OAA), and [PSA](https://github.com/jiwoon-ahn/psa))
  - Please see [the official deeplab website](http://liangchiehchen.com/projects/DeepLabv2_resnet.html) for information
  ```
  CRF parameters: bi_w = 4, bi_xy_std = 67, bi_rgb_std = 3, pos_w = 3, pos_xy_std = 1.
  ```
### Segmentation network
- We utilize [DeepLab-V2](https://arxiv.org/abs/1606.00915) 
  for the segmentation network. 
- Please see [deeplab-pytorch](https://github.com/kazuto1011/deeplab-pytorch) for the implementation in PyTorch.
- We used the pretrained model for VGG16 based network from [DeepLab official](http://liangchiehchen.com/projects/DeepLab_Models.html) and for ResNet101-based network from [OAA official](https://github.com/PengtaoJiang/OAA).
  
## Results


![results](figure/effect_EPS.png)

## Acknowledgement
This code is highly borrowed from [PSA](https://github.com/jiwoon-ahn/psa). Thanks to Jiwoon, Ahn.
