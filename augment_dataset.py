import argparse
import random
from pathlib import Path

import cv2
import numpy as np

from coin_counter_pipeline import augment_coin, ensure_color, list_image_files


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="硬幣影像資料擴增工具")
    parser.add_argument("--source-dir", type=Path, required=True, help="原始影像資料夾")
    parser.add_argument("--output-dir", type=Path, required=True, help="擴增後影像輸出資料夾")
    parser.add_argument("--augmentations-per-image", type=int, default=20, help="每張影像生成的擴增數量")
    parser.add_argument("--include-original", action="store_true", help="輸出時保留原始影像")
    parser.add_argument("--seed", type=int, default=42, help="亂數種子")
    parser.add_argument("--suffix", type=str, default=".jpg", help="輸出影像副檔名，例如 .jpg 或 .png")
    return parser.parse_args()


def save_image(image: np.ndarray, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.suffix.lower() in {".jpg", ".jpeg"}:
        cv2.imwrite(str(destination), image, [cv2.IMWRITE_JPEG_QUALITY, 95])
    else:
        cv2.imwrite(str(destination), image)


def augment_directory(
    source_dir: Path,
    output_dir: Path,
    augmentations_per_image: int,
    include_original: bool,
    suffix: str,
) -> None:
    image_paths = list(list_image_files(source_dir))
    if not image_paths:
        raise ValueError(f"在 {source_dir} 找不到任何影像。")
    output_dir.mkdir(parents=True, exist_ok=True)
    for image_path in image_paths:
        image = cv2.imread(str(image_path))
        if image is None:
            print(f"跳過無法讀取的影像: {image_path}")
            continue
        image = ensure_color(image)
        stem = image_path.stem
        if include_original:
            save_image(image, output_dir / f"{stem}_orig{suffix}")
        augmented_images = augment_coin(image, augmentations_per_image)
        for idx, aug in enumerate(augmented_images, start=1):
            save_image(aug, output_dir / f"{stem}_aug_{idx:03d}{suffix}")


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    augment_directory(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        augmentations_per_image=args.augmentations_per_image,
        include_original=args.include_original,
        suffix=args.suffix,
    )
    print(f"資料擴增完成，結果已輸出至 {args.output_dir}")


if __name__ == "__main__":
    main()
