import argparse
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np
import sys

from coin_counter_pipeline import count_coins, load_model
from coin_counter_pipeline import CuMLPredictWrapper  # 需要和訓練時同一實作
   
sys.modules["__main__"].CuMLPredictWrapper = CuMLPredictWrapper


LABEL_ORDER_DEFAULT = ["1", "5", "10", "50"]


def read_image_list(list_file: Path) -> List[str]:
    lines = [line.strip() for line in list_file.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return []
    try:
        total = int(lines[0])
        entries = lines[1:]
        if total != len(entries):
            print(f"警告: {list_file} 宣告 {total} 筆，但實際找到 {len(entries)} 行，會以實際行數為準。")
        return entries
    except ValueError:
        return lines


def read_counts_file(path: Path) -> Dict[str, List[int]]:
    lines = [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        return {}
    try:
        total = int(lines[0])
        rows = lines[1:]
    except ValueError:
        total = len(lines)
        rows = lines
    counts: Dict[str, List[int]] = {}
    for idx, row in enumerate(rows):
        parts = [int(value) for value in row.split()]
        counts[str(idx)] = parts
    if total != len(rows):
        print(f"警告: {path} 宣告 {total} 筆，但實際有 {len(rows)} 行。")
    return counts


def write_counts_file(path: Path, counts: Sequence[Sequence[int]]) -> None:
    lines = [str(len(counts))]
    lines.extend(" ".join(str(value) for value in row) for row in counts)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def gather_predictions(
    model_path: Path,
    image_dir: Path,
    image_list: List[str],
    image_size: int,
    label_order: Sequence[str],
) -> Tuple[List[List[int]], List[str]]:
    model, encoder = load_model(model_path)
    label_order = list(label_order) if label_order else list(encoder.classes_)
    image_paths = []
    for name in image_list:
        candidate = image_dir / name
        if candidate.exists():
            image_paths.append(candidate)
        else:
            print(f"警告: 找不到 {candidate}，略過。")
    counts_map = count_coins(model, image_paths, (image_size, image_size), label_order)
    results = []
    valid_names = []
    for path in image_paths:
        summary = counts_map.get(path.name, {})
        row = [summary.get(label, 0) for label in label_order]
        results.append(row)
        valid_names.append(path.name)
    return results, valid_names


def evaluate(predictions: Sequence[Sequence[int]], ground_truth: Sequence[Sequence[int]]) -> Tuple[float, List[float]]:
    per_image = []
    for pred, gt in zip(predictions, ground_truth):
        gt_sum = sum(gt)
        if gt_sum == 0:
            score = 1.0 if sum(pred) == 0 else 0.0
        else:
            diff = sum(abs(p - g) for p, g in zip(pred, gt))
            score = max(0.0, 1.0 - diff / gt_sum)
        per_image.append(score)
    if not per_image:
        return 0.0, []
    return float(np.mean(per_image)), per_image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="硬幣計數結果產生與（可選）評估")
    parser.add_argument("--image-dir", type=Path, required=True, help="影像資料夾")
    parser.add_argument("--image-list", type=Path, required=True, help="in.txt 檔案位置")
    parser.add_argument(
        "--ground-truth",
        type=Path,
        required=False,
        help="gt.txt 檔案位置（可省略，省略則只產生 out.txt 不計分）",
    )
    parser.add_argument("--model-path", type=Path, required=True, help="已訓練模型 coin_svm.joblib")
    parser.add_argument("--output-file", type=Path, default=Path("out.txt"), help="輸出 out.txt 檔案位置")
    parser.add_argument("--image-size", type=int, default=128, help="切圖片時的輸入尺寸")
    parser.add_argument(
        "--label-order",
        type=str,
        nargs="+",
        default=LABEL_ORDER_DEFAULT,
        help="輸出欄位順序 (預設為 1 5 10 50)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_list = read_image_list(args.image_list)
    if not image_list:
        print("未從 image-list 取得任何檔名。")
        return
    predictions, valid_names = gather_predictions(
        args.model_path,
        args.image_dir,
        image_list,
        args.image_size,
        args.label_order,
    )
    if not predictions:
        print("沒有成功產生任何預測。")
        return
    write_counts_file(args.output_file, predictions)
    print(f"已輸出預測至 {args.output_file}")
    if args.ground_truth:
        gt_counts_map = read_counts_file(args.ground_truth)
        ground_truth_rows = []
        for idx, name in enumerate(valid_names):
            key = str(idx)
            counts = gt_counts_map.get(key)
            if counts is None:
                print(f"警告: gt.txt 未提供第 {idx} 筆 ({name}) 的標註，改填 0。")
                counts = [0] * len(args.label_order)
            ground_truth_rows.append(counts)
        accuracy, per_image = evaluate(predictions, ground_truth_rows)
        print("各影像得分:")
        for name, score in zip(valid_names, per_image):
            print(f"{name}: {score:.4f}")
        print(f"整體 accuracy: {accuracy:.4f}")
    else:
        print("未提供 ground-truth，跳過評分。")


if __name__ == "__main__":
    main()
