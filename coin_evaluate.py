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
    CROP_SCALE_INFERENCE,
    is_valid_coin_crop,
)

sys.modules["__main__"].CuMLPredictWrapper = CuMLPredictWrapper
LABEL_ORDER_DEFAULT = ["1", "5", "10", "50"]

def read_image_list(list_file: Path) -> List[str]:
    lines = [line.strip() for line in list_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines: return []
    try:
        int(lines[0])
        return lines[1:]
    except ValueError:
        return lines

def read_counts_file(path: Path) -> Dict[str, List[int]]:
    if not path.exists(): return {}
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

def draw_results(image: np.ndarray, circles: List[Tuple[int, int, int, str]]) -> np.ndarray:
    canvas = image.copy()
    h, w = canvas.shape[:2]
    font_scale = max(0.5, min(w, h) / 1000.0) 
    thickness = max(1, int(min(w, h) / 500.0))
    for x, y, r, label in circles:
        cv2.circle(canvas, (x, y), r, (0, 255, 0), 2)
        cv2.circle(canvas, (x, y), 2, (0, 0, 255), 3)
        text = f"${label}"
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        text_x = x - tw // 2
        text_y = y - r - 10 if y - r - 10 > th else y 
        cv2.rectangle(canvas, (text_x - 2, text_y - th - 2), (text_x + tw + 2, text_y + 2), (0, 0, 0), -1)
        cv2.putText(canvas, text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 255, 255), thickness)
    return canvas

def evaluate_and_visualize(model_path: Path, image_dir: Path, image_list: List[str], image_size: int, label_order: Sequence[str], debug_dir: Path = None) -> Tuple[List[List[int]], List[str]]:
    model, encoder = load_model(model_path)
    label_order = list(label_order) if label_order else list(encoder.classes_)
    results, valid_names = [], []
    
    if debug_dir:
        debug_dir.mkdir(parents=True, exist_ok=True)
        print(f"🕵️  視覺化除錯模式已啟用，結果將存至: {debug_dir}")

    for name in image_list:
        path = image_dir / name
        if not path.exists():
            print(f"警告: 找不到 {path}，略過。")
            continue
        image = cv2.imread(str(path))
        if image is None: continue
        image = ensure_color(image)
        
        circles = detect_coins(image)
        circles = dedup_circles(circles, image.shape)
        
        coin_counts = {label: 0 for label in label_order}
        detected_info = [] 
        for circle in circles:
            crop = crop_coin(image, circle, scale=CROP_SCALE_INFERENCE)
            if not is_valid_coin_crop(crop): continue 
            
            resized_crop = preprocess_image(crop, (image_size, image_size))
            features = extract_features(resized_crop)
            prediction = to_numpy(model.predict([features]))[0]
            label = encoder.inverse_transform([prediction])[0]
            if label in coin_counts: coin_counts[label] += 1
            detected_info.append((circle[0], circle[1], circle[2], label))

        if debug_dir:
            debug_img = draw_results(image, detected_info)
            summary_text = " | ".join([f"${k}:{v}" for k, v in coin_counts.items()])
            h, w = debug_img.shape[:2]
            info_scale = max(0.5, min(w, h) / 1000.0)
            cv2.putText(debug_img, summary_text, (10, int(h*0.05)), cv2.FONT_HERSHEY_SIMPLEX, info_scale, (0, 0, 255), 2)
            cv2.imwrite(str(debug_dir / f"debug_{name}"), debug_img)

        results.append([coin_counts[label] for label in label_order])
        valid_names.append(name)
        if debug_dir: print(f"  Processed {name}: 偵測到 {len(detected_info)} 枚硬幣")

    return results, valid_names

def evaluate_metrics(predictions: Sequence[Sequence[int]], ground_truth: Sequence[Sequence[int]]) -> Tuple[float, List[float]]:
    per_image = []
    for pred, gt in zip(predictions, ground_truth):
        gt_sum = sum(gt)
        if gt_sum == 0: score = 1.0 if sum(pred) == 0 else 0.0
        else:
            diff = sum(abs(p - g) for p, g in zip(pred, gt))
            score = max(0.0, 1.0 - diff / gt_sum)
        per_image.append(score)
    return (float(np.mean(per_image)), per_image) if per_image else (0.0, [])

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="硬幣計數評估與視覺化工具")
    parser.add_argument("--image-dir", type=Path, default=Path("coin_datas"), help="影像資料夾")
    parser.add_argument("--image-list", type=Path, default=Path("in.txt"), help="in.txt")
    parser.add_argument("--ground-truth", type=Path, help="gt.txt (選填)")
    parser.add_argument("--model-path", type=Path, default=Path("coin_svm_v7.joblib"), help="模型路徑")
    parser.add_argument("--output-file", type=Path, default=Path("out.txt"), help="輸出結果路徑")
    parser.add_argument("--image-size", type=int, default=112, help="影像尺寸 (需與訓練一致)")
    parser.add_argument("--debug-dir", type=Path, default=Path("debug_images"), help="[選填] 指定資料夾以輸出標記後的除錯圖片")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    image_list = read_image_list(args.image_list)
    if not image_list:
        print("❌ 錯誤: in.txt 沒有內容")
        return
    predictions, valid_names = evaluate_and_visualize(args.model_path, args.image_dir, image_list, args.image_size, LABEL_ORDER_DEFAULT, debug_dir=args.debug_dir)
    if not predictions:
        print("❌ 沒有產生任何預測")
        return
    write_counts_file(args.output_file, predictions)
    print(f"✅ 預測結果已寫入 {args.output_file}")
    if args.ground_truth and args.ground_truth.exists():
        gt_map = read_counts_file(args.ground_truth)
        gt_rows = [gt_map.get(str(i), [0]*4) for i in range(len(valid_names))]
        acc, scores = evaluate_metrics(predictions, gt_rows)
        print(f"\n📊 整體 Accuracy: {acc:.4f}")
    else:
        print("\n⚠️ 未提供 gt.txt 或檔案不存在，跳過評分。")

if __name__ == "__main__":
    main()