import argparse
import random
from pathlib import Path
from typing import Iterable, List

import cv2
import numpy as np


RANDOM_SEED = 42


def list_images(directory: Path) -> Iterable[Path]:
    for path in sorted(directory.iterdir()):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            yield path


def ensure_color(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def rotate(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def translate(image: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def perspective(image: np.ndarray, ratio: float = 0.08) -> np.ndarray:
    h, w = image.shape[:2]
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    offset_x = ratio * w
    offset_y = ratio * h
    dst = np.float32(
        [
            [random.uniform(0, offset_x), random.uniform(0, offset_y)],
            [w - 1 - random.uniform(0, offset_x), random.uniform(0, offset_y)],
            [random.uniform(0, offset_x), h - 1 - random.uniform(0, offset_y)],
            [w - 1 - random.uniform(0, offset_x), h - 1 - random.uniform(0, offset_y)],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array([(idx / 255.0) ** inv_gamma * 255 for idx in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


def blur(image: np.ndarray) -> np.ndarray:
    k = random.choice([3, 5, 7])
    return cv2.GaussianBlur(image, (k, k), 0)


def add_noise(image: np.ndarray, std: float) -> np.ndarray:
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def random_shadow(image: np.ndarray, opacity: float = 0.4) -> np.ndarray:
    h, w = image.shape[:2]
    mask = np.zeros((h, w), dtype=np.uint8)
    x1, y1 = random.randint(0, w), random.randint(0, h)
    x2, y2 = random.randint(0, w), random.randint(0, h)
    thickness = random.randint(max(5, w // 10), max(10, w // 4))
    cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=thickness)
    mask = cv2.GaussianBlur(mask, (51, 51), 0)
    alpha = (mask.astype(np.float32) / 255.0 * opacity)[..., None]
    return (image.astype(np.float32) * (1 - alpha)).astype(np.uint8)


def augment(image: np.ndarray) -> np.ndarray:
    result = image.copy()
    if random.random() < 0.7:
        result = rotate(result, random.uniform(-35, 35))
    if random.random() < 0.6:
        shift_x = random.randint(-int(0.07 * result.shape[1]), int(0.07 * result.shape[1]))
        shift_y = random.randint(-int(0.07 * result.shape[0]), int(0.07 * result.shape[0]))
        result = translate(result, shift_x, shift_y)
    if random.random() < 0.5:
        result = perspective(result)
    if random.random() < 0.6:
        result = adjust_gamma(result, random.uniform(0.6, 1.6))
    if random.random() < 0.5:
        result = blur(result)
    if random.random() < 0.4:
        result = add_noise(result, std=random.uniform(5, 20))
    if random.random() < 0.3:
        result = random_shadow(result, opacity=random.uniform(0.25, 0.55))
    return result


def augment_file(image_path: Path, output_dir: Path, copies: int) -> List[Path]:
    image = cv2.imread(str(image_path))
    if image is None:
        return []
    image = ensure_color(image)
    saved = []
    cv2.imwrite(str(output_dir / image_path.name), image)
    saved.append(output_dir / image_path.name)
    stem = image_path.stem
    suffix = image_path.suffix
    for idx in range(1, copies + 1):
        aug = augment(image)
        output_path = output_dir / f"{stem}_aug{idx:02d}{suffix}"
        cv2.imwrite(str(output_path), aug)
        saved.append(output_path)
    return saved


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="針對硬幣影像資料集進行資料擴增")
    parser.add_argument("--input-dir", type=Path, required=True, help="原始影像資料夾")
    parser.add_argument("--output-dir", type=Path, required=True, help="擴增影像輸出資料夾")
    parser.add_argument("--copies", type=int, default=10, help="每張影像額外產生的擴增張數")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    total = 0
    for image_path in list_images(args.input_dir):
        saved = augment_file(image_path, args.output_dir, args.copies)
        total += len(saved)
        print(f"{image_path.name} -> {len(saved)} 張")
    print(f"總計輸出 {total} 張影像於 {args.output_dir}")


if __name__ == "__main__":
    main()
