import argparse
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np

from coin_counter_pipeline import crop_coin, detect_coins, ensure_color, list_image_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="使用 HoughCircles 擷取硬幣並輸出裁切小圖")
    parser.add_argument("--input-dir", type=Path, required=True, help="原始影像資料夾")
    parser.add_argument("--output-dir", type=Path, required=True, help="裁切後硬幣輸出資料夾")
    parser.add_argument("--dp", type=float, default=1.2, help="Hough 圓檢 dp 參數，預設 1.2")
    parser.add_argument("--min-dist-ratio", type=float, default=0.12, help="圓心最小距離比例，預設 0.12")
    parser.add_argument("--param1", type=float, default=120, help="Canny 高閾值，預設 120")
    parser.add_argument("--param2", type=float, default=40, help="圓檢門檻，越低越寬鬆，預設 40")
    parser.add_argument("--min-radius-ratio", type=float, default=0.08, help="半徑下限比例，預設 0.08")
    parser.add_argument("--max-radius-ratio", type=float, default=0.35, help="半徑上限比例，預設 0.35")
    parser.add_argument("--suffix", type=str, default=".jpg", help="輸出影像副檔名，預設 .jpg")
    parser.add_argument(
        "--crop-scale",
        type=float,
        default=1.10,
        help="裁切時在半徑外再放大的比例，避免邊緣被截斷，預設 1.10",
    )
    parser.add_argument("--median-ksize", type=int, default=5, help="圓檢前的中值濾波核，須為奇數；0 表示跳過")
    parser.add_argument(
        "--clahe",
        action="store_true",
        help="啟用 CLAHE 對比增強 (對光照不均常有幫助)",
    )
    parser.add_argument(
        "--param2-scan",
        type=float,
        nargs="+",
        help="多組 param2 嘗試並合併結果，例如: --param2-scan 32 36 40",
    )
    parser.add_argument(
        "--dedup-dist-ratio",
        type=float,
        default=0.06,
        help="合併多輪偵測的重疊圓，圓心距離低於此比例(乘以短邊)視為同一顆硬幣。",
    )
    return parser.parse_args()


def save_crop(crop: cv2.UMat, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(destination), crop)


def preprocess_for_circles(image: np.ndarray, median_ksize: int, use_clahe: bool) -> np.ndarray:
    processed = image
    if median_ksize and median_ksize % 2 == 1:
        processed = cv2.medianBlur(processed, median_ksize)
    if use_clahe:
        lab = cv2.cvtColor(processed, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        processed = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    return processed


def dedup_circles(circles: Sequence[Tuple[int, int, int]], image_shape: Tuple[int, int, int], dist_ratio: float) -> List[Tuple[int, int, int]]:
    if not circles:
        return []
    min_dim = min(image_shape[:2])
    dist_thresh = max(1, int(min_dim * dist_ratio))
    kept: List[Tuple[int, int, int]] = []
    for cx, cy, r in circles:
        is_dup = False
        for kx, ky, kr in kept:
            if (cx - kx) ** 2 + (cy - ky) ** 2 <= dist_thresh**2:
                is_dup = True
                break
        if not is_dup:
            kept.append((cx, cy, r))
    return kept


def crop_with_scale(image: np.ndarray, circle: Tuple[int, int, int], scale: float) -> np.ndarray:
    x, y, r = circle
    r_scaled = int(r * scale)
    x1 = max(x - r_scaled, 0)
    y1 = max(y - r_scaled, 0)
    x2 = min(x + r_scaled, image.shape[1])
    y2 = min(y + r_scaled, image.shape[0])
    return image[y1:y2, x1:x2]


def extract_coins_from_image(
    image_path: Path,
    output_dir: Path,
    dp: float,
    min_dist_ratio: float,
    param1: float,
    param2: float,
    min_radius_ratio: float,
    max_radius_ratio: float,
    suffix: str,
    median_ksize: int,
    use_clahe: bool,
    param2_scan: Sequence[float] | None,
    dedup_dist_ratio: float,
    crop_scale: float,
) -> int:
    image = cv2.imread(str(image_path))
    if image is None:
        print(f"跳過無法讀取的影像: {image_path}")
        return 0
    image = ensure_color(image)
    processed = preprocess_for_circles(image, median_ksize, use_clahe)
    param2_values: Iterable[float] = param2_scan if param2_scan else [param2]
    all_circles: List[Tuple[int, int, int]] = []
    for p2 in param2_values:
        circles = detect_coins(
            processed,
            dp=dp,
            min_dist_ratio=min_dist_ratio,
            param1=param1,
            param2=p2,
            min_radius_ratio=min_radius_ratio,
            max_radius_ratio=max_radius_ratio,
        )
        all_circles.extend(circles)
    circles = dedup_circles(all_circles, image.shape, dedup_dist_ratio)
    saved = 0
    for idx, circle in enumerate(circles, start=1):
        crop = crop_with_scale(image, circle, crop_scale)
        if crop.size == 0:
            continue
        filename = f"{image_path.stem}_coin{idx:02d}{suffix}"
        save_crop(crop, output_dir / filename)
        saved += 1
    return saved


def extract_dataset(
    input_dir: Path,
    output_dir: Path,
    dp: float,
    min_dist_ratio: float,
    param1: float,
    param2: float,
    min_radius_ratio: float,
    max_radius_ratio: float,
    suffix: str,
    median_ksize: int,
    use_clahe: bool,
    param2_scan: Sequence[float] | None,
    dedup_dist_ratio: float,
    crop_scale: float,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    image_paths: List[Path] = list(list_image_files(input_dir))
    if not image_paths:
        print(f"{input_dir} 中沒有找到影像。")
        return
    total_saved = 0
    for image_path in image_paths:
        saved = extract_coins_from_image(
            image_path,
            output_dir,
            dp,
            min_dist_ratio,
            param1,
            param2,
            min_radius_ratio,
            max_radius_ratio,
            suffix,
            median_ksize,
            use_clahe,
            param2_scan,
            dedup_dist_ratio,
            crop_scale,
        )
        total_saved += saved
        print(f"{image_path.name}: 擷取 {saved} 張")
    print(f"完成！總計輸出 {total_saved} 張裁切硬幣至 {output_dir}")


def main() -> None:
    args = parse_args()
    extract_dataset(
        input_dir=args.input_dir,
        output_dir=args.output_dir,
        dp=args.dp,
        min_dist_ratio=args.min_dist_ratio,
        param1=args.param1,
        param2=args.param2,
        min_radius_ratio=args.min_radius_ratio,
        max_radius_ratio=args.max_radius_ratio,
        suffix=args.suffix,
        median_ksize=args.median_ksize,
        use_clahe=args.clahe,
        param2_scan=args.param2_scan,
        dedup_dist_ratio=args.dedup_dist_ratio,
        crop_scale=args.crop_scale,
    )


if __name__ == "__main__":
    main()

