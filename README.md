# 硬幣辨識與計數流程說明

此專案協助以傳統影像特徵 (HOG + LBP) 搭配 SVM 完成 1、5、10、50 元硬幣的分類與計數，並提供資料擴增、推論及評估工具。以下整理必要環境、資料結構、操作步驟與評估方式，讓你可以依照簡報的規格快速完成整體流程。

---

## 1. 環境需求

請使用 Python 3.9 以上版本，並安裝下列套件：

```bash
pip install opencv-python numpy scikit-image scikit-learn joblib
```

> 若使用 Windows PowerShell，可改成 `python -m pip install ...` 以避免多個 Python 版本衝突。

---

## 2. 資料夾與檔案結構

```
final_project/
├─ coin_counter_pipeline.py      # 訓練、推論與硬幣計數主流程
├─ coin_dataset_augment.py       # 單張硬幣影像資料擴增工具
├─ coin_evaluate.py              # 依簡報格式產生 out.txt 並計算 accuracy
├─ generate_in_file.py           # 從影像資料夾自動生成 in.txt
├─ 錢幣預測試資料庫/             # 原始測試影像 (001.jpg ~ 010.jpg)
├─ 錢幣預測試資料庫_augmented/   # 透過擴增腳本產生的影像
├─ in.txt                        # 依簡報規格列出的影像清單 (可重新生成)
└─ coin_svm.joblib               # 訓練後模型 (執行訓練流程後產生)
```

> 若要自行整理訓練資料，請另行建立 `train_coins/1`, `train_coins/5`, `train_coins/10`, `train_coins/50` 等子資料夾。

---

## 3. 操作流程總覽

1. **HoughCircles 裁切硬幣**：先把原始 10 張影像的硬幣全部挖出來。
2. **手動分類裁切圖**：將小圖拖進 `train_coins/1, 5, 10, 50`。
3. **特徵提取 + 訓練**：使用 HOG + LBP 特徵搭配 SVM 訓練。
4. **生成 in.txt**：配合評估格式列出要推論的影像清單。
5. **產生 out.txt 與評估**：使用訓練好的模型計數，並與 gt.txt 比較得到 accuracy。

各步驟詳細說明如下。

---

## 3.1 新增：Hough 圓檢裁切 → 手動分類 → 訓練流程

### Step 1：裁切硬幣小圖 (`coin_extract_circles.py`)
使用 OpenCV 的 Hough Circle Transform 從整張影像擷取硬幣並存成個別小圖：
```bash
python coin_extract_circles.py ^
  --input-dir "final_project/錢幣預測試資料庫" ^
  --output-dir "final_project/cropped_coins" ^
  --param2 38 ^
  --min-radius-ratio 0.07 ^
  --max-radius-ratio 0.40
```
- 可依需求微調 `--param2`（越低越寬鬆）、`--min-radius-ratio`、`--max-radius-ratio`、`--min-dist-ratio`。
- 若偵測不穩：可加 `--clahe`（光照增強）、`--median-ksize 5`（雜訊平滑）、或用多組 `--param2-scan 32 36 40` 並透過 `--dedup-dist-ratio` 合併重疊圓。
- 若裁切常截邊：調高 `--crop-scale`（預設 1.10，建議 1.10~1.20）放大裁切框，減少缺角。
- 每張輸出檔名格式：`原檔名_coinXX.jpg`，方便對照來源。

### Step 2：手動分類
將 `cropped_coins` 內的小圖拖到 `train_coins/1`, `train_coins/5`, `train_coins/10`, `train_coins/50` 四個資料夾中，確認標籤無誤。

### Step 3：特徵提取
`coin_counter_pipeline.py` 內的 `extract_features` 會先將硬幣裁切圖 resize 為灰階，再取：
- HOG：orientations=9, pixels_per_cell=(16,16), cells_per_block=(2,2), L2-Hys。
- LBP：P=24, R=3，直方圖正規化後與 HOG 向量串接。

### Step 4：訓練（HOG+LBP + SVM）
使用手動分好類別的 `train_coins` 直接訓練，預設使用 RBF kernel 並含類別平衡、標準化：
```bash
python coin_counter_pipeline.py ^
  --train-dir "final_project/train_coins" ^
  --augmentations-per-image 30 ^
  --model-path "final_project/coin_svm.joblib"
```
- 若只想用小圖訓練，不指定 `--predict-dir`，即可純訓練保存模型。
- 也可改用 `--skip-train` 搭配既有模型進行推論。

> 後續的 in.txt 生成與 out.txt 評估仍沿用原本方式（見下方第 6、7 節）。

---

## 4. 資料擴增 (`coin_dataset_augment.py`)

針對資料夾內的每張影像進行旋轉、平移、透視、亮度調整、模糊、雜訊與陰影等隨機變換，以模擬各種拍攝情境。每張影像預設會輸出原圖 + 20 種擴增版本 (可調整)。

```bash
python coin_dataset_augment.py ^
  --input-dir "final_project/錢幣預測試資料庫" ^
  --output-dir "final_project/錢幣預測試資料庫_augmented" ^
  --copies 20
```

執行後終端機會顯示每張影像產生的張數，總結算也會顯示於最後。

---

## 5. 訓練與推論 (`coin_counter_pipeline.py`)

`coin_counter_pipeline.py` 會：

- 自動對訓練資料做幾何與光度擴增。
- 轉換灰階影像後擷取 HOG + LBP 特徵。
- 使用 SVM (RBF kernel) 搭配標準化與類別平衡進行訓練。
- 提供交叉驗證與驗證集評估結果。
- 對指定資料夾進行硬幣偵測、裁切並分類面額，輸出每張影像的面額統計。

### 5.1 訓練並推論

```bash
python coin_counter_pipeline.py ^
  --train-dir "train_coins" ^
  --predict-dir "final_project/錢幣預測試資料庫_augmented" ^
  --augmentations-per-image 30 ^
  --model-path "coin_svm.joblib"
```

- `--train-dir`：依面額建好的訓練資料夾根目錄。
- `--predict-dir`：推論時要統計硬幣的影像資料夾。
- `--augmentations-per-image`：每張訓練影像額外生成的擴增張數。
- `--model-path`：模型儲存路徑 (會產生 `.joblib` 檔)。

指令結束後會列出交叉驗證與驗證集的 Precision / Recall / F1，並儲存模型，同時在推論階段印出各影像的面額統計結果。

### 5.2 僅推論既有模型

```bash
python coin_counter_pipeline.py ^
  --predict-dir "final_project/錢幣預測試資料庫_augmented" ^
  --model-path "coin_svm.joblib" ^
  --skip-train
```

加入 `--skip-train` 後會載入既有模型，不再重新訓練。

---

## 6. 生成 in.txt (`generate_in_file.py`)

配合簡報需求，`in.txt` 需在第一行填入影像總數，其餘每行為影像檔名。本腳本預設只列出原始檔案 (忽略檔名含 `_aug` 的影像)，確保格式與範例一致。

```bash
python generate_in_file.py ^
  --image-dir "final_project/錢幣預測試資料庫_augmented" ^
  --output-file "final_project/in.txt"
```

若需要包含擴增影像，可改用：

```bash
python generate_in_file.py ^
  --image-dir "final_project/錢幣預測試資料庫_augmented" ^
  --output-file "final_project/in_aug.txt" ^
  --lowercase --include-augmented
```

> `--lowercase` 視評測系統是否區分大小寫而定，預設保持原樣。

---

## 7. 產生 out.txt 與評估 (`coin_evaluate.py`)

此腳本會：

1. 讀取 `in.txt` 取得要推論的影像名稱。
2. 載入 `coin_svm.joblib`，對影像進行硬幣偵測與面額分類。
3. 依序寫入 `out.txt` (第一行為筆數，其餘為各面額數量)。
4. 與 `gt.txt` (真實標註) 比較，計算每張影像得分與整體 accuracy，公式與簡報一致。

```bash
python coin_evaluate.py ^
  --image-dir "final_project/錢幣預測試資料庫_augmented" ^
  --image-list "final_project/in.txt" ^
  --ground-truth "final_project/gt.txt" ^
  --model-path "final_project/coin_svm.joblib" ^
  --output-file "final_project/out.txt"
```

輸出範例：

```
已輸出預測至 final_project/out.txt
各影像得分:
001.jpg: 1.0000
002.jpg: 0.8500
...
整體 accuracy: 0.9300
```

請先依評估需求準備 `gt.txt`：第一行為樣本數，後續每行列出 `[1元 5元 10元 50元]` 的實際硬幣數量，例如：

```
5
0 1 2 1
1 1 3 0
1 1 1 2
1 2 3 1
1 0 1 1
```

---

## 8. 推薦的測試流程

1. 清空或備份舊的擴增資料夾。
2. 重新執行 `coin_dataset_augment.py` 產生新影像。
3. 確認訓練資料已整理好，執行 `coin_counter_pipeline.py` 進行訓練。
4. 使用 `generate_in_file.py` 更新 `in.txt`。
5. 準備或更新 `gt.txt`，再透過 `coin_evaluate.py` 產生 `out.txt` 並檢查 accuracy。
6. 視結果調整擴增數量、SVM 參數或硬幣偵測門檻，反覆迭代。

---

## 9. 常見問題

- **`cv2.imread` 回傳 None**：確認影像路徑是否正確、檔名大小寫是否一致，以及是否含有非 ASCII 字元 (如有，建議改用英數檔名)。
- **模型分類效果不佳**：可嘗試增加訓練資料、提高 `--augmentations-per-image`、調整 HOG 參數，或檢查 `gt.txt` 是否與硬幣實際數量一致。
- **硬幣偵測不到**：調整 `detect_coins` 中 HoughCircles 的參數，例如 `param2` 或 `minRadius/maxRadius`；也可先預處理影像（如增強對比）。

---

## 10. 後續延伸方向

- 加入更多手動標註的偽幣樣本，搭配度量學習或異常偵測增強辨識能力。
- 以 Mask R-CNN、YOLOv8-seg 等 instance segmentation 方法改善堆疊硬幣偵測。
- 將目前流程包裝成圖形介面或腳本參數檔，降低操作門檻。

---

快去試試吧！如果有新的靈感或想微調流程，直接修改腳本參數就可以持續優化，堅持下去你的硬幣辨識系統會越來越完美 ❤️
