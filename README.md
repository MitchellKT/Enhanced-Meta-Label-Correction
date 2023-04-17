# Enhanced Meta Label Correction (EMLC)

![](https://github.com/iccv23anonymous/Enhanced-Meta-Label-Correction/blob/main/Teacher.jpg?raw=true)

## Description :information_source:
This is an official PyTorch implementation of the "Enhanced Meta Label Correction for Coping with Label Corruption" paper.

  * [Description](#description-information_source)
  * [Results](#results-chart_with_upwards_trend)
  * [Running the Code](#running-the-code-runner)
    + [Training](#training-weight_lifting)
  * [Dependencies](#dependencies-floppy_disk)
  * [Credits](#credits-copyright)
  
  ## Results :chart_with_upwards_trend:

CIFAR-10 with 1k clean samples:

|Method/Noise |  20% | 50% | 80% | 90% | Asym. 40% | 
|:--- |:---:|:---:|:---:|:---:|:---:|
|MLC|92.6 |88.1 |77.4 |67.9 |–|
|Previous SOTA|93.4 |90.07 | 86.78 | 79.52 | 91.6 |
|EMLC|**93.64** |**92.23**|**90.97**|**90.13**|**92.71**|

CIFAR-100 with 1k clean samples:

|Method/Noise |  20% | 50% | 80% | 90% |
|:--- |:---:|:---:|:---:|:---:|
|MLC|66.5 | 52.4 | 18.9 | 14.2 |
|Previous SOTA| 72.5 | 65.4 | 55.22 | 16.7 | 
|EMLC|**73.27** |**68.98**|**59.34**|**53.78**|

Clothing1M:

|Method | Accuracy |
|:--- |:---:|
|MLC|75.78|
|Previous SOTA| 77.83 |
|EMLC|**78.70** |

  
## Running The Code :runner:

Before training the models please:
1. Put the datasets in the `data` sub-directory.
2. For the CIFAR experiments, put the self-supervised pretrained models in the `pretrained` sub-directory. You may use [SupContrast](https://github.com/HobbitLong/SupContrast "SupContrast") implementation for training the SSL models.
3. For the Clothing experiment, run the preprocessing procedure:
```
python CLOTHING1M/load_cloth1m_data.py --root=[path_to_clothing_dataset]
```
4. Go over the [dependencies](#dependencies-floppy_disk).

### Training :weight_lifting:

To start training on the Clothing1M dataset run:

``` 
python main_clothing.py --gpuid=a --n_gpus=b
```

To start training on the CIFAR datasets run:

``` 
python main_clothing.py --gpuid=a --n_gpus=b --dataset=c --corruption_type=d --corruption_level=e -ssl_path=f
```

The parameters and their meaning:

|  Parameter |  Description | Type | Possible values |
|:---:| :---:|:---:|:---:|
|`gpuid` |The id of the primary GPU to train on |int | [0,#GPUS) |
|`n_gpus` |Number of GPUs to train on |int |[1,#GPUS] |
|`dataset`|CIFAR dataset to train on |string | Either `cifar10` or `cifar100` |
|`corruption_type`|Type of artifical noise |string|`unif` for symmetric noise injection and `flip` for asymmetric noise injection|
|`corruption_level`|Injected Noise level |float|(0,1)|
|`ssl_path`|Path to the relevant SSL pretrained model |string|  `pretrained/*.pth`  |


## Dependencies :floppy_disk:

**Before trying to run anything please ensure that the packages below are up to date.**

|Library         |  Minimal Version |
| :---         |     :---:      |
|`NumPy`|  1.23.5 |
|`PyTorch`|   1.13.1 |
|`torchvision`|   0.14.1 |
|`tensorboard`|   2.11.2 |
|`argparse`| 1.1 |
|`tqdm`| 4.64.0 |

## Credits :copyright:
This repository is heavily based on [MLC](https://github.com/microsoft/MLC "MLC's repository").
