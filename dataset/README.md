# Dataset preparation

公開 repository 不包含完整原始圖片資料。請在專案根目錄建立以下結構：

```text
picture/
├── Aloe/
├── Zingiber/
├── Mentha/
├── Perilla/
└── Turmeric/
```

每個類別資料夾放入對應的 `.jpg`、`.jpeg` 或 `.png` 圖片，接著執行 `python split_dataset.py`，程式會建立：

```text
picture/
├── train/<class-name>/
└── test/<class-name>/
```

`crawler.py` 使用 iNaturalist API 下載圖片；若要重新擷取，請先確認來源資料的授權條件，並在程式中設定要下載的類別。完整資料集不隨作品集 repository 發布。
