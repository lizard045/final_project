import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import cv2
import numpy as np
import sys

# 引用 pipeline 的功能
from coin_counter_pipeline import (
    load_model,
    detect_coins,
    dedup_circles,
    crop_coin,
    preprocess_image,
    extract_features,
    to_numpy,
    ensure_color,
    CuMLPredictWrapper,
    DEDUP_DIST_RATIO,
    DEDUP_DIST_RATIO_HEAVY,
    DEDUP_HEAVY_THRESHOLD,
    CROP_SCALE_INFERENCE
)

# 注入 CuML wrapper 避免 pickle 錯誤
sys.modules["__main__"].CuMLPredictWrapper = CuMLPredictWrapper

LABEL_ORDER_DEFAULT = ["1", "5", "10", "50"]

def read_image_list(list_file: Path) -> List[str]:
    lines = [line.strip() for line in list_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return []
    # 處理第一行可能是數量的格式
    try:
        int(lines[0])
        return lines[1:]
    except ValueError:
        return lines

def read_counts_file(path: Path) -> Dict[str, List[int]]:
    if not path.exists():
        return {}
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    counts: Dict[str, List[int]] = {}
    start_idx = 1 if len(lines) > 0 and lines[0].isdigit() else 0
    for idx, row in enumerate(lines[start_idx:]):
        parts = [int(value) for value in row.split()]
        counts[str(idx)] = parts
    return counts

def write_counts_file(path: Path, counts: Sequence[Sequence[int]]) -> None:
    lines = [str(len(counts))]
    lines.extend(" ".join(str(value) for value in row) for row in counts)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

def draw_results(
    image: np.ndarray, 
    circles: List[Tuple[int, int, int, str]]
) -> np.ndarray:
    """在影像上繪製偵測結果 (圓圈 + 文字)"""
    canvas = image.copy()
    for x, y, r, label in circles:
        # 畫圓 (綠色)
        cv2.circle(canvas, (x, y), r, (0, 255, 0), 2)
        # 畫圓心
        cv2.circle(canvas, (x, y), 2, (0, 0, 255), 3)
        # 畫標籤文字 (紅色，帶黑邊)
        text = f"${label}"
        font_scale = max(0.6, r / 40.0)
        thickness = max(1, int(r / 20.0))
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        # 文字背景
        cv2.rectangle(canvas, (x - tw//2 - 2, y - 5 - th - 2), (x + tw//2 + 2, y - 5 + 2), (0, 0, 0), -1)
        # 文字本體
        cv2.putText(canvas, text, (x - tw//2, y - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
    return canvas

def evaluate_and_visualize(
    model_path: Path,
    image_dir: Path,
    image_list: List[str],
    image_size: int,
    label_order: Sequence[str],
    debug_dir: Path = None,
) -> Tuple[List[List[int]], List[str]]:
    
    model, encoder = load_model(model_path)
    label_order = list(label_order) if label_order else list(encoder.classes_)
    
    results = []
    valid_names = []
    
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"🕵️  視覺化除錯模式已啟用，結果將存至: {debug_dir}")

    for name in image_list:
        path = image_dir / name
        if not path.exists():
            print(f"警告: 找不到 {path}，略過。")
            continue
            
        # 1. 讀取影像
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = ensure_color(image)
        
        # 2. 偵測硬幣 (使用 pipeline 邏輯)
        circles = detect_coins(image)  # 這裡會呼叫 pipeline 內含的多重掃描
        circles = dedup_circles(circles, image.shape, dist_ratio=DEDUP_DIST_RATIO)
        
        # 二次過濾邏輯 (與 pipeline 保持一致)
        if len(circles) > DEDUP_HEAVY_THRESHOLD:
            circles = dedup_circles(circles, image.shape, dist_ratio=DEDUP_DIST_RATIO_HEAVY)
            if circles:
                radii = np.asarray([r for _, _, r in circles])
                median_r = np.median(radii)
                lower = int(max(1, 0.65 * median_r))
                upper = int(1.45 * median_r)
                circles = [(x, y, r) for x, y, r in circles if lower <= r <= upper]

        # 3. 預測每個硬幣的面額
        coin_counts = {label: 0 for label in label_order}
        detected_info = [] # 儲存 (x, y, r, label) 供繪圖用
        
        for circle in circles:
            crop = crop_coin(image, circle, scale=CROP_SCALE_INFERENCE)
            if crop.size == 0:
                continue
                
            gray = preprocess_image(crop, (image_size, image_size))
            features = extract_features(gray)
            
            # 預測
            prediction = to_numpy(model.predict([features]))[0]
            label = encoder.inverse_transform([prediction])[0]
            
            # 記錄
            if label in coin_counts:
                coin_counts[label] += 1
            detected_info.append((circle[0], circle[1], circle[2], label))

        # 4. 輸出視覺化結果
        if debug_dir:
            debug_img = draw_results(image, detected_info)
            # 在左上角印出總計
            summary_text = " | ".join([f"${k}:{v}" for k, v in coin_counts.items()])
            cv2.putText(debug_img, summary_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
            cv2.imwrite(str(debug_dir / f"debug_{name}"), debug_img)

        # 5. 整理結果
        row = [coin_counts[label] for label in label_order]
        results.append(row)
        valid_names.append(name)
        
        if debug_dir:
            print(f"  Processed {name}: 偵測到 {len(detected_info)} 枚硬幣")

    return results, valid_names

def evaluate_metrics(predictions: Sequence[Sequence[int]], ground_truth: Sequence[Sequence[int]]) -> Tuple[float, List[float]]:
    per_image = []
    for pred, gt in zip(predictions, ground_truth):
        gt_sum = sum(gt)
        if gt_sum == 0:
            score = 1.0 if sum(pred) == 0 else 0.0
        else:
            diff = sum(abs(p - g) for p, g in zip(pred, gt))
            score = max(0.0, 1.0 - diff / gt_sum)
        per_image.append(score)
    return (float(np.mean(per_image)), per_image) if per_image else (0.0, [])

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="硬幣計數評估與視覺化工具")
    parser.add_argument("--image-dir", type=Path, required=True, help="影像資料夾")
    parser.add_argument("--image-list", type=Path, required=True, help="in.txt")
    parser.add_argument("--ground-truth", type=Path, help="gt.txt (選填)")
    parser.add_argument("--model-path", type=Path, default=Path("coin_svm.joblib"), help="模型路徑")
    parser.add_argument("--output-file", type=Path, default=Path("out.txt"), help="輸出結果路徑")
    parser.add_argument("--image-size", type=int, default=112, help="影像尺寸 (需與訓練一致)")
    # 新增 debug 參數
    parser.add_argument("--debug-dir", type=Path, help="[選填] 指定資料夾以輸出標記後的除錯圖片")
    
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    image_list = read_image_list(args.image_list)
    
    if not image_list:
        print("❌ 錯誤: in.txt 沒有內容")
        return

    predictions, valid_names = evaluate_and_visualize(
        args.model_path,
        args.image_dir,
        image_list,
        args.image_size,
        LABEL_ORDER_DEFAULT,
        debug_dir=args.debug_dir  # 傳入 debug 路徑
    )

    if not predictions:
        print("❌ 沒有產生任何預測")
        return

    write_counts_file(args.output_file, predictions)
    print(f"✅ 預測結果已寫入 {args.output_file}")

    if args.ground_truth and args.ground_truth.exists():
        gt_map = read_counts_file(args.ground_truth)
        gt_rows = []
        for i, name in enumerate(valid_names):
            # 嘗試用行號或檔名找 GT，這裡簡化用行號
            key = str(i) 
            if key not in gt_map:
                gt_rows.append([0]*4)
            else:
                gt_rows.append(gt_map[key])
        
        acc, scores = evaluate_metrics(predictions, gt_rows)
        print(f"\n📊 整體 Accuracy: {acc:.4f}")
    else:
        print("\n⚠️ 未提供 gt.txt 或檔案不存在，跳過評分。")

if __name__ == "__main__":
    main()