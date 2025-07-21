# PHA-CMX: Hybrid Attention for RGB-X Semantic Segmentation
本專案為一個多模態語義分割模型 **HA-CMX (Hybrid Attention-CMX)**，專為處理 **RGB-X**（如 RGB-D、RGB-T）場景中的語義分割任務所設計。該模型融合多種注意力機制，強化模態互補性與語義對齊能力，在多種資料集（如 NYUv2、MFNet）上展現優異性能。

## 🧠 模型架構簡介

PHA-CMX 結合以下三大核心模組：

- **Dynamic Cross-Attention (DCA)**：進行雙模態互補特徵交叉學習，並透過 Gate MLP 控制信息通道。
- **Parallel Hybrid Attention (PHA)**：整合 CoordAttention 與 ShiftViTBlockv2，強化低階與中階特徵的語義引導。
  



### 📁 Datasets結構
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
### 🚀 安裝與環境需求
``` 
Python ≥ 3.7
PyTorch ≥ 1.10
CUDA ≥ 11.1
``` 
安裝依賴套件：
 ``` 
pip install -r requirements.txt
 ``` 
### 🏋️‍♀️ 訓練流程 請使用以下指令開始訓練：

預訓練 
[pretrained segformer](https://drive.google.com/drive/folders/10XgSW8f7ghRs9fJ0dE-EV8G2E_guVsT5).
 ``` 
bash run.sh
 ``` 


 ``` 
### 📊 評估流程 執行以下指令可自動針對每 25 個 epoch 從第 300 開始評估：
 ``` 
python eval.py -e log_XXXX_mit_XX/checkpoint/epoch-XXX.ph -d 0
 ``` 
### 🧪 支援資料集
 ``` 
 NYUv2

 MFNet

 PST900

 SUNGRGBD
 ``` 
### 🧪數據
``` 
| 模型名稱   | 參數量 (M) | FLOPs (G) | mIoU (%) |
|------------|------------|-----------|----------|
| CMX-B2     | 28.7       | 70.3      | 58.2     |
| PHA-CMX-B2 | 29.1       | 72.8      | 59.7     |
| PHA-CMX-B4 | 63.2       | 139.4     | 61.3     |


### 🙌 貢獻者（Contributors） 主作者：Sheng（Project Maintainer） 指導單位：淡江大學電機系RVL Chi-Yi Tsai

### 📚 致謝與引用（Credits）
本專案基於以下開源專案進行改寫與擴充：

- [RGBX_Semantic_Segmentation (by huaaaliu)](https://github.com/huaaaliu/RGBX_Semantic_Segmentation)
感謝原作者釋出代碼，作為本模型的基礎架構參考。
