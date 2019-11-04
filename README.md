# Pytorch Material Classification

This repo provides examples for material classification in GTOS, GTOS-MOBILE, DTD and MINC dataset using PyTorch.

## Setup

### Prerequisites

- Ubuntu
- Pytorch 
  - `pip3 install torch torchvision`
- Easydict
  - `pip3 install easydict`  
- tqdm
  - `pip3 install tqdm`  
- [Pytorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding) 
  - `pip3 install torch-encoding`
  - Note: You need to install Pytorch 1.0 for torch-encoding, or you can modify the encoding layer based on [this](https://github.com/zhanghang1989/PyTorch-Encoding/issues/161) for latest Pytorch.

### Getting Started

- Clone this repo:
```bash
git clone git@github.com:jiaxue1993/pytorch-material-classification.git
cd pytorch-material-classification/
``` 

- Download [GTOS](https://1drv.ms/u/s!AmTf4gl42ObnbXCS4GrRUAWutWI?e=u7nRrR), [GTOS_MOBILE](https://1drv.ms/u/s!AmTf4gl42ObnblEtikrw4HfD9fc?e=LjJir4), [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz), [MINC](http://opensurfaces.cs.cornell.edu/static/minc/minc-2500.tar.gz) to the dataset folder

- Navigate to different experiment folder and train network. For example, you can finetune ResNet on GTOS-MOBILE dataset with followint command
```bash
cd experiments/gtos_mobile.finetune.resnet/
python train.py
```

## Accuracy & Statistics
Coming soon

## Citation

Please consider citing following projects in your publications if it helps your research.

**Differential Angular Imaging for Material Recognition** [[pdf]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Xue_Differential_Angular_Imaging_CVPR_2017_paper.pdf)
```
@inproceedings{xue2017differential,
  title={Differential angular imaging for material recognition},
  author={Xue, Jia and Zhang, Hang and Dana, Kristin and Nishino, Ko},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={764--773},
  year={2017}
}
```

**Deep Texture Manifold for Ground Terrain Recognition** [[pdf]](http://openaccess.thecvf.com/content_cvpr_2018/papers/Xue_Deep_Texture_Manifold_CVPR_2018_paper.pdf)
```
@inproceedings{xue2018deep,
  title={Deep texture manifold for ground terrain recognition},
  author={Xue, Jia and Zhang, Hang and Dana, Kristin},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={558--567},
  year={2018}
}
```

**Deep TEN: Texture Encoding Network** [[pdf]](http://openaccess.thecvf.com/content_cvpr_2017/papers/Zhang_Deep_TEN_Texture_CVPR_2017_paper.pdf)
```
@inproceedings{zhang2017deep,
  title={Deep ten: Texture encoding network},
  author={Zhang, Hang and Xue, Jia and Dana, Kristin},
  booktitle={Proceedings of the IEEE conference on computer vision and pattern recognition},
  pages={708--717},
  year={2017}
}
```

## Acknowledgement
Part of the code comes from [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding), [TorchSeg](https://github.com/ycszen/TorchSeg), [pytorch-mobilenet-v2](https://github.com/tonylins/pytorch-mobilenet-v2)
