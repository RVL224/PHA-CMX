# PHA-CMX: Hybrid Attention for RGB-X Semantic Segmentation

This repo implements a multimodal semantic segmentation model, **PHA-CMX (Parallel Hybrid Attention-CMX)**, specifically designed for semantic segmentation tasks in **RGB-X** (such as RGB-D and RGB-T) scenes. This model integrates multiple attention mechanisms to enhance modal complementarity and semantic alignment capabilities, demonstrating excellent performance on various datasets (such as SUNGRGBD and MFNet).
  
Related research findings have been published in the following papers:

Chi-Yi Tsai and Dao-Sheng Du, "PHA-CMX: Parallel Hybrid Attention with Dynamic Cross-Attention for Multimodal Semantic Segmentation," Under review.

Please cite this article if you use our work in your research.
  
## 🧠 Model Architecture Introduction

PHA-CMX combines the following two core modules:

- **Dynamic Feature Fusion (DFF)**：Performs bimodal complementary feature cross-learning and controls the information channels through GateMLP.
- **Parallel Hybrid Attention (PHA)**：Integrates CoordAttention and ShiftViTBlockv2 to enhance the semantic guidance of low- and mid-level features.
  
### 📁 Datasets structure
 ``` 
<datasets>
|-- <DatasetName1>
   |-- <RGBFolder>
       |-- <name1>.<ImageFormat>
       |-- <name2>.<ImageFormat>
       ...
   |-- <ModalXFolder>
       |-- <name1>.<ModalXFormat>
       |-- <name2>.<ModalXFormat>
       ...
   |-- <LabelFolder>
       |-- <name1>.<LabelFormat>
       |-- <name2>.<LabelFormat>
       ...
   |-- train.txt
   |-- test.txt
|-- <DatasetName2>
|-- ...
 ``` 
### 🚀 Installation and Environmental Requirements
``` 
Python ≥ 3.7
PyTorch ≥ 1.10
CUDA ≥ 11.1
``` 

### 🏋️‍♀️ Training Procedure - Please use the following instructions to begin training:

Pre-training 
[pretrained segformer](https://drive.google.com/drive/folders/10XgSW8f7ghRs9fJ0dE-EV8G2E_guVsT5).

 ``` 
bash run.sh
 ``` 
### 📊 Evaluation process:
 ``` 
python eval.py -e log_XXXX_mit_XX/checkpoint/epoch-XXX.pth -d 0
 ``` 
### 🧪 Datasets:
 [MFNet](https://www.mi.t.u-tokyo.ac.jp/static/projects/mil_multispectral/)
 [PST900](https://github.com/ShreyasSkandanS/pst900_thermal_rgb)
 [SUNGRGBD](https://rgbd.cs.princeton.edu/)
 
### 🧪Experimental data
#### MFNet
| Model        | Backbone | mIoU (%) |
|--------------|----------|----------|
| CMX-B2       | MiT-B2   | 58.2     |
| PHA-CMX-B2   | MiT-B2   | 59.7     |
| PHA-CMX-B4   | MiT-B4   | 61.3     |
#### SUNRGBD
| Model        | Backbone | mIoU (%) |
|--------------|----------|----------|
| CMX-B2       | MiT-B2   | 49.7     |
| PHA-CMX-B2   | MiT-B2   | 50.8     |

### 📚 Acknowledgments and Citations（Credits）
This repo is based on and expanded upon the following open-source project:

- [RGBX_Semantic_Segmentation (by huaaaliu)](https://github.com/huaaaliu/RGBX_Semantic_Segmentation)
Thanks to the original author for releasing the code as the basic architecture reference for this model.
