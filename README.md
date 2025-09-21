# Visual Semantic Description Generation with MLLMs for Image-Text Matching

![Static Badge](https://img.shields.io/badge/Pytorch-EE4C2C)
![License: MIT](https://img.shields.io/badge/License-Apache%202.0-yellow.svg)

The codes for our paper "Visual Semantic Description Generation with MLLMs for Image-Text Matching(VSD)", ,which is accepted by the ICME2025. We referred to the implementations of [GPO](https://github.com/woodfrog/vse_infty), [HREM](https://github.com/CrossmodalGroup/HREM), and [MiniCPM-V](https://github.com/OpenBMB/MiniCPM-V) to build up our codes. We express our gratitude for these outstanding works.

## Introduction

Image-text matching (ITM) aims to address the fundamental challenge of aligning visual and textual modalities, which inherently differ in their representations—continuous, high-dimensional image features vs. discrete, structured text. We propose a novel framework that bridges the modality gap by leveraging multimodal large language models (MLLMs) as visual semantic parsers. By generating rich Visual Semantic Descriptions (VSD), MLLMs provide semantic anchor that facilitate cross-modal alignment. 
Our approach combines: (1) Instance-level alignment by fusing visual features with VSD to enhance the linguistic expressiveness of image representations, and (2) Prototype-level alignment through VSD clustering to ensure category-level consistency. These modules can be seamlessly integrated into existing ITM models. Extensive experiments on Flickr30K and MSCOCO demonstrate substantial performance improvements. The approach also exhibits remarkable zero-shot generalization to cross-domain tasks, including news and remote sensing ITM.

![overview](https://github.com/Image-Text-Matching/VSD/blob/main/overview.png)

## Performance

![main_result](https://github.com/Image-Text-Matching/VSD/blob/main/main_result.png)

![cross_domin_result](https://github.com/Image-Text-Matching/VSD/blob/main/cross_domin_result.png)

**Note**: We have open-sourced the complete implementation code for GPO+VSD⋆, GPO+VSD†, HREM+VSD⋆, and HREM+VSD†. For the CLIP+VSD⋆ and CLIP+VSD† versions, researchers can reproduce the results based on the technical details in our published paper and the already open-sourced related code. As the structure and organization of this portion of code require further optimization, we plan to release it after completing the code refactoring.

## Preparation

### Environments

We recommended the following dependencies.

- Python 3.9
- [PyTorch](http://pytorch.org/) 1.11
- transformers  4.36.0
- open-clip-torch 2.24.0
- numpy 1.23.5
- tensorboard-logger 0.1.0
- The specific required environment can be found [here](https://github.com/Image-Text-Matching/AAHR/AAHR/blob/main/requirements.txt)


### Data

All data sets used in the experiment and the necessary external components are organized in the following manner:

```
data
├── coco
│   ├── precomp  # pre-computed BUTD region features for COCO, provided by SCAN
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── train_aux_cap_bge_cpm_full.npy
│   │      ├── train_aux_cap_bge_flor_det.npy
│   │      ├── train_cap_bge.npy
│   │      ├── testall_aux_cap_bge_cpm_full_1.npy
│   │      ├── testall_aux_cap_bge_flor_det.npy
│   │      ├── testall_cap_bge.npy
│   │      ├── ......
│   │
│   │── id_mapping.json
│   ├── images   # (option) raw coco images for OpenCLIP
│        ├── train2014
│        └── val2014
│  
├── f30k
│   ├── precomp  # pre-computed BUTD region features for Flickr30K, provided by SCAN
│   │      ├── train_ids.txt
│   │      ├── train_caps.txt
│   │      ├── train_aux_cap_bge_cpm_full.npy
│   │      ├── train_aux_cap_bge_flor_det.npy
│   │      ├── train_cap_bge.npy
│   │      ├── test_aux_cap_bge_cpm_full_1.npy
│   │      ├── test_aux_cap_bge_flor_det.npy
│   │      ├── ......
│   │
│   │── id_mapping.json
│   ├── images   # (option) raw flickr30k images for OpenCLIP
│          ├── xxx.jpg
│          └── ...
│   
└── vocab  # vocab files provided by SCAN (only used when the text backbone is BiGRU)

VSD
├── bert-base-uncased    # the pretrained checkpoint files for BERT-base
│   ├── config.json
│   ├── tokenizer_config.txt
│   ├── vocab.txt
│   ├── pytorch_model.bin
│   ├── ......

└── CLIP                         # (option) the pretrained checkpoint files for OpenCLIP
│   ├── config.json
│   ├── tokenizer_config.json
│   ├── vocab.json
│   ├── open_clip_config.json
│   ├── open_clip_pytorch_model.bin
│   ├── ......
│  
└── ....

```

#### Data Sources:

- Visual semantics describe preprocessed features: [Baidu Yun](https://pan.baidu.com/s/1ClRpz4akDOnTZlCYrS_Blw?pwd=EVDP) (code: EVDP)
- BUTD features: [SCAN (Kaggle)](https://www.kaggle.com/datasets/kuanghueilee/scan-features)  or [Baidu Yun](https://pan.baidu.com/s/1Dmnf0q9J29m4-fyL7ubqdg?pwd=AAHR) (code: AAHR)
- MSCOCO images: [Official](https://cocodataset.org/#download)  or [Paddle Paddle](https://aistudio.baidu.com/datasetdetail/28191)
- Flickr30K images: [Official](https://shannon.cs.illinois.edu/DenotationGraph/) or [SCAN (Kaggle)](https://www.kaggle.com/datasets/eeshawn/flickr30k?select=flickr30k_images)
- Pretrained models: [BERT-base-uncased](https://huggingface.co/google-bert/bert-base-uncased) , [MiniCPM-V 2.6](https://huggingface.co/openbmb/MiniCPM-V-2_6) , [Florence-2-large-ft](https://huggingface.co/microsoft/Florence-2-large-ft) ,[bge-large-en-v1.5](https://huggingface.co/BAAI/bge-large-en-v1.5) and [OpenCLIP](https://huggingface.co/laion/CLIP-ViT-B-32-laion2B-s34B-b79K) from HuggingFace

## Training

Train MSCOCO and Flickr30K from scratch:

```
bash  run_f30k.sh
```

```
bash  run_coco.sh
```

## Evaluation

Modify the corresponding parameters in eval.py to test the Flickr30K or MSCOCO data set:

```
python eval.py  --dataset f30k  --data_path "path/to/dataset"
```

```
python eval.py  --dataset coco --data_path "path/to/dataset"
```

##  Citation

If you find our paper and code useful in your research, please consider giving a star ⭐ and a citation 📝:

```
@inproceedings{chen2025VSD,
  title={Visual Semantic Description Generation with MLLMs for Image-Text Matching},
  author={Chen, Junyu and Gao, Yihua and Li, Mingyong},
  booktitle={2025 IEEE International Conference on Multimedia and Expo (ICME)},
  pages={1--6},
  year={2025},
  organization={IEEE}
}
```
