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
git clone https://github.com/jiaxue1993/pytorch-material-classification.git
cd pytorch-material-classification/
``` 

- Download [GTOS](https://1drv.ms/u/s!AmTf4gl42ObncLmEnEv4R5LyxT4?e=ekkFfX), [GTOS_MOBILE](https://1drv.ms/u/s!AmTf4gl42ObnblEtikrw4HfD9fc?e=LjJir4), [DTD](https://www.robots.ox.ac.uk/~vgg/data/dtd/download/dtd-r1.0.1.tar.gz), [MINC](http://opensurfaces.cs.cornell.edu/static/minc/minc-2500.tar.gz) to the dataset folder

- Navigate to different experiment folder and train network. For example, you can finetune ResNet on GTOS-MOBILE dataset with followint command
```bash
cd experiments/gtos_mobile.finetune.resnet/
python train.py
```

## Accuracy & Statistics


<table class="tg">
<thead>
  <tr>
    <th class="tg-5unb">Base Model</th>
    <th class="tg-5unb">Dataset</th>
    <th class="tg-5unb">Method</th>
    <th class="tg-5unb">Accuracy</th>
    <th class="tg-5unb">Pretrained Model<br></th>
  </tr>
</thead>
<tbody>
  <tr>
    <td class="tg-lcl0" rowspan="4">MobileNet</td>
    <td class="tg-wp8o" rowspan="4"><span style="font-weight:normal;font-style:normal;text-decoration:none">GTOS</span></td>
    <td class="tg-wp8o">Finetune</td>
    <td class="tg-wp8o">80.4</td>
    <td class="tg-wp8o" rowspan="6"><a href="https://1drv.ms/u/s!AmTf4gl42Obncc3EohCJubNVHZQ?e=PZ1eAC" target="_blank" rel="noopener noreferrer">One Drive</a></td>
  </tr>
  <tr>
    <td class="tg-wp8o">DAIN</td>
    <td class="tg-wp8o">82.5</td>
  </tr>
  <tr>
    <td class="tg-wp8o">DEP</td>
    <td class="tg-wp8o">83.3</td>
  </tr>
  <tr>
    <td class="tg-wp8o">TEAN</td>
    <td class="tg-wp8o">84.7</td>
  </tr>
  <tr>
    <td class="tg-lcl0" rowspan="2">ResNet-50</td>
    <td class="tg-wp8o">DTD</td>
    <td class="tg-wp8o"><span style="font-weight:normal;font-style:normal;text-decoration:none">DEP</span></td>
    <td class="tg-wp8o">73.2</td>
  </tr>
  <tr>
    <td class="tg-wp8o">MINC</td>
    <td class="tg-wp8o"><span style="font-weight:normal;font-style:normal;text-decoration:none">DEP</span></td>
    <td class="tg-wp8o">82.0</td>
  </tr>
</tbody>
</table>

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

**Differential Viewpoints for Ground Terrain Material Recognition** [[pdf]](https://ieeexplore.ieee.org/abstract/document/9200748)[[arxiv]](https://arxiv.org/pdf/2009.11072.pdf)
```
@article{xue2020differential,
  title={Differential Viewpoints for Ground Terrain Material Recognition},
  author={Xue, Jia and Zhang, Hang and Nishino, Ko and Dana, Kristin},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2020},
  publisher={IEEE}
}
```

## Acknowledgement
Part of the code comes from [PyTorch-Encoding](https://github.com/zhanghang1989/PyTorch-Encoding), [TorchSeg](https://github.com/ycszen/TorchSeg), [pytorch-mobilenet-v2](https://github.com/tonylins/pytorch-mobilenet-v2)
