# Herb Image Recognition

以深度學習辨識五種中藥材／植物影像的研究型專案。這個 repository 專注於圖片資料擷取、資料集切分、模型訓練、評估，以及 Tkinter 圖形介面的單張圖片預測。

## Highlights

- 從 iNaturalist API 擷取植物影像
- 以 80/20 比例建立 train/test dataset
- 比較自架 `HerbCNN` 與 `MobileNetV2`
- 輸出 confusion matrix、訓練紀錄與預測信心度
- 提供可直接操作的中文 Tkinter 圖形介面

目前分類：`Aloe`、`Zingiber`、`Mentha`、`Perilla`、`Turmeric`。

## Repository layout

```text
.
├── crawler.py             # 從 iNaturalist 下載圖片
├── split_dataset.py       # 建立 picture/train 與 picture/test
├── train.py              # MobileNetV2 訓練
├── train2.py             # HerbCNN 訓練
├── evaluate.py           # MobileNetV2 評估
├── evaluate2.py          # HerbCNN 評估
├── predict.py            # Tkinter 單張圖片辨識介面
├── models/               # 已訓練模型權重
├── results/              # 評估圖表與訓練紀錄
└── dataset/              # 資料集準備說明
```

## Quick start

```bash
python -m venv .venv
# Windows PowerShell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

將圖片依照 `dataset/README.md` 放入 `picture/`，再執行：

```bash
python split_dataset.py
python train.py       # MobileNetV2
python train2.py      # HerbCNN
python evaluate.py
python evaluate2.py
python predict.py
```

GPU 會在 PyTorch 可用時自動使用；沒有 GPU 時會退回 CPU。

## Public-repository note

原始圖片資料集、虛擬環境、IDE 設定與大型模型權重沒有放入公開 repository。這樣可以避免不必要的大型檔案與來源授權問題；資料來源與資料夾格式請見 [`dataset/README.md`](dataset/README.md)。

## License and attribution

程式碼為研究與作品展示用途。若重新下載 iNaturalist 圖片，請依照各筆觀測資料與 iNaturalist 的授權要求進行署名及使用。
