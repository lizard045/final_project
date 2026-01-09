import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import joblib
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC


RANDOM_SEED = 42
LBP_POINTS = 24
LBP_RADIUS = 3
LBP_BINS = LBP_POINTS + 2


def list_image_files(root: Path) -> Iterable[Path]:
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
            yield path


def ensure_color(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        return cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    if image.shape[2] == 4:
        return cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)
    return image


def rotate_image(image: np.ndarray, angle: float) -> np.ndarray:
    h, w = image.shape[:2]
    center = (w // 2, h // 2)
    matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
    return cv2.warpAffine(image, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)


def scale_image(image: np.ndarray, scale: float) -> np.ndarray:
    h, w = image.shape[:2]
    resized = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
    if scale >= 1.0:
        start_y = (resized.shape[0] - h) // 2
        start_x = (resized.shape[1] - w) // 2
        return resized[start_y:start_y + h, start_x:start_x + w]
    canvas = np.zeros_like(image)
    canvas[:resized.shape[0], :resized.shape[1]] = resized
    return canvas


def translate_image(image: np.ndarray, shift_x: int, shift_y: int) -> np.ndarray:
    h, w = image.shape[:2]
    matrix = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
    return cv2.warpAffine(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def perspective_transform(image: np.ndarray, max_shift_ratio: float = 0.08) -> np.ndarray:
    h, w = image.shape[:2]
    shift_x = max_shift_ratio * w
    shift_y = max_shift_ratio * h
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    dst = np.float32(
        [
            [random.uniform(0, shift_x), random.uniform(0, shift_y)],
            [w - 1 - random.uniform(0, shift_x), random.uniform(0, shift_y)],
            [random.uniform(0, shift_x), h - 1 - random.uniform(0, shift_y)],
            [w - 1 - random.uniform(0, shift_x), h - 1 - random.uniform(0, shift_y)],
        ]
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(image, matrix, (w, h), borderMode=cv2.BORDER_REFLECT_101)


def adjust_gamma(image: np.ndarray, gamma: float) -> np.ndarray:
    inv_gamma = 1.0 / max(gamma, 1e-6)
    table = np.array([(idx / 255.0) ** inv_gamma * 255 for idx in np.arange(256)]).astype("uint8")
    return cv2.LUT(image, table)


def add_gaussian_noise(image: np.ndarray, std: float) -> np.ndarray:
    noise = np.random.normal(0, std, image.shape).astype(np.float32)
    noisy = np.clip(image.astype(np.float32) + noise, 0, 255)
    return noisy.astype(np.uint8)


def random_shadow(image: np.ndarray, opacity: float = 0.4) -> np.ndarray:
    h, w = image.shape[:2]
    x1, y1 = random.randint(0, w), random.randint(0, h)
    x2, y2 = random.randint(0, w), random.randint(0, h)
    mask = np.zeros((h, w), dtype=np.uint8)
    cv2.line(mask, (x1, y1), (x2, y2), 255, thickness=random.randint(w // 8, w // 3))
    blurred = cv2.GaussianBlur(mask, (51, 51), 0)
    alpha = blurred.astype(np.float32) / 255.0 * opacity
    shadow = image.astype(np.float32) * (1 - alpha[..., None])
    return shadow.astype(np.uint8)


def augment_coin(image: np.ndarray, count: int) -> List[np.ndarray]:
    augmented = []
    for _ in range(count):
        variant = image.copy()
        if random.random() < 0.7:
            variant = rotate_image(variant, random.uniform(-35, 35))
        if random.random() < 0.5:
            variant = scale_image(variant, random.uniform(0.85, 1.15))
        if random.random() < 0.5:
            shift_x = random.randint(-int(0.08 * variant.shape[1]), int(0.08 * variant.shape[1]))
            shift_y = random.randint(-int(0.08 * variant.shape[0]), int(0.08 * variant.shape[0]))
            variant = translate_image(variant, shift_x, shift_y)
        if random.random() < 0.4:
            variant = perspective_transform(variant)
        if random.random() < 0.6:
            variant = adjust_gamma(variant, random.uniform(0.6, 1.6))
        if random.random() < 0.4:
            variant = add_gaussian_noise(variant, std=random.uniform(5, 18))
        if random.random() < 0.3:
            variant = random_shadow(variant, opacity=random.uniform(0.25, 0.55))
        augmented.append(variant)
    return augmented


def preprocess_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    resized = cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(resized, cv2.COLOR_BGR2GRAY)
    return gray


def extract_features(gray_image: np.ndarray) -> np.ndarray:
    hog_features = hog(
        gray_image,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        transform_sqrt=True,
        feature_vector=True,
    )
    lbp = local_binary_pattern(gray_image, LBP_POINTS, LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp, bins=LBP_BINS, range=(0, LBP_BINS))
    hist = hist.astype(np.float32)
    hist_sum = hist.sum() + 1e-6
    hist /= hist_sum
    return np.concatenate([hog_features, hist])


def build_dataset(
    train_dir: Path,
    target_size: Tuple[int, int],
    augmentations_per_image: int,
) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    images = []
    labels = []
    classes = sorted({path.parent.name for path in list_image_files(train_dir)})
    if not classes:
        raise ValueError(f"未在 {train_dir} 找到任何圖像，請確認資料夾內依面值分子資料夾。")
    for class_name in classes:
        class_dir = train_dir / class_name
        for image_path in list_image_files(class_dir):
            image = cv2.imread(str(image_path))
            if image is None:
                continue
            base = ensure_color(image)
            gray = preprocess_image(base, target_size)
            images.append(extract_features(gray))
            labels.append(class_name)
            aug_images = augment_coin(base, augmentations_per_image)
            for aug in aug_images:
                aug_gray = preprocess_image(aug, target_size)
                images.append(extract_features(aug_gray))
                labels.append(class_name)
    return np.asarray(images), np.asarray(labels), classes


def train_model(
    features: np.ndarray,
    labels: np.ndarray,
    cross_validation_folds: int,
) -> Tuple[Pipeline, Dict[str, float], str, np.ndarray]:
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    pipeline = Pipeline(
        [
            ("scaler", StandardScaler(with_mean=False)),
            ("svm", SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_SEED)),
        ]
    )
    x_train, x_valid, y_train, y_valid = train_test_split(
        features,
        encoded_labels,
        test_size=0.2,
        random_state=RANDOM_SEED,
        stratify=encoded_labels,
    )
    pipeline.fit(x_train, y_train)
    y_valid_pred = pipeline.predict(x_valid)
    y_valid_labels = encoder.inverse_transform(y_valid)
    y_valid_pred_labels = encoder.inverse_transform(y_valid_pred)
    report_dict = classification_report(
        y_valid_labels,
        y_valid_pred_labels,
        labels=encoder.classes_,
        target_names=encoder.classes_,
        zero_division=0,
        output_dict=True,
    )
    report_text = classification_report(
        y_valid_labels,
        y_valid_pred_labels,
        labels=encoder.classes_,
        target_names=encoder.classes_,
        digits=4,
        zero_division=0,
    )
    conf_mat = confusion_matrix(
        y_valid_labels,
        y_valid_pred_labels,
        labels=encoder.classes_,
    )
    skf = StratifiedKFold(n_splits=cross_validation_folds, shuffle=True, random_state=RANDOM_SEED)
    cv_scores = []
    for train_idx, test_idx in skf.split(features, encoded_labels):
        pipeline.fit(features[train_idx], encoded_labels[train_idx])
        score = pipeline.score(features[test_idx], encoded_labels[test_idx])
        cv_scores.append(score)
    pipeline.fit(features, encoded_labels)
    metrics = {
        "validation_accuracy": float(report_dict["accuracy"]),
        "validation_precision": float(report_dict["macro avg"]["precision"]),
        "validation_recall": float(report_dict["macro avg"]["recall"]),
        "validation_f1": float(report_dict["macro avg"]["f1-score"]),
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_std": float(np.std(cv_scores)),
    }
    pipeline.named_steps["label_encoder"] = encoder  # type: ignore[attr-defined]
    return pipeline, metrics, report_text, conf_mat


def detect_coins(
    image: np.ndarray,
    dp: float = 1.2,
    min_dist_ratio: float = 0.12,
    param1: float = 120,
    param2: float = 40,
    min_radius_ratio: float = 0.08,
    max_radius_ratio: float = 0.35,
) -> List[Tuple[int, int, int]]:
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (9, 9), 2)
    min_dim = min(image.shape[:2])
    min_dist = max(20, int(min_dim * min_dist_ratio))
    min_radius = int(min_dim * min_radius_ratio)
    max_radius = int(min_dim * max_radius_ratio)
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=dp,
        minDist=min_dist,
        param1=param1,
        param2=param2,
        minRadius=min_radius,
        maxRadius=max_radius,
    )
    if circles is None:
        return []
    return [(int(x), int(y), int(r)) for x, y, r in circles[0]]


def crop_coin(image: np.ndarray, circle: Tuple[int, int, int]) -> np.ndarray:
    x, y, r = circle
    x1 = max(x - r, 0)
    y1 = max(y - r, 0)
    x2 = min(x + r, image.shape[1])
    y2 = min(y + r, image.shape[0])
    return image[y1:y2, x1:x2]


def count_coins(
    model: Pipeline,
    image_paths: Sequence[Path],
    target_size: Tuple[int, int],
    label_order: Sequence[str],
) -> Dict[str, Dict[str, int]]:
    encoder: LabelEncoder = model.named_steps["label_encoder"]  # type: ignore[assignment]
    counts: Dict[str, Dict[str, int]] = {}
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None:
            continue
        image = ensure_color(image)
        circles = detect_coins(image)
        coin_counter: Counter[str] = Counter()
        for circle in circles:
            crop = crop_coin(image, circle)
            if crop.size == 0:
                continue
            gray = preprocess_image(crop, target_size)
            features = extract_features(gray)
            prediction = model.predict([features])[0]
            label = encoder.inverse_transform([prediction])[0]
            coin_counter[label] += 1
        ordered_counts = {label: coin_counter.get(label, 0) for label in label_order}
        counts[path.name] = ordered_counts
    return counts


def save_model(model: Pipeline, destination: Path) -> None:
    model_copy = Pipeline(model.steps[:-1] + [("svm", model.named_steps["svm"])])
    encoder = model.named_steps["label_encoder"]
    payload = {"model": model_copy, "encoder": encoder}
    joblib.dump(payload, destination)


def load_model(path: Path) -> Tuple[Pipeline, LabelEncoder]:
    payload = joblib.load(path)
    model: Pipeline = payload["model"]
    encoder: LabelEncoder = payload["encoder"]
    model.named_steps["label_encoder"] = encoder  # type: ignore[attr-defined]
    return model, encoder


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="硬幣分類與計數 (HOG + LBP + SVM)")
    parser.add_argument("--train-dir", type=Path, required=True, help="訓練資料夾，需依面值建立子資料夾")
    parser.add_argument("--predict-dir", type=Path, help="要進行硬幣計數的影像資料夾")
    parser.add_argument("--model-path", type=Path, default=Path("coin_svm.joblib"), help="模型儲存路徑")
    parser.add_argument("--image-size", type=int, default=128, help="特徵提取時的影像邊長")
    parser.add_argument("--augmentations-per-image", type=int, default=20, help="每張原圖生成的擴增數量")
    parser.add_argument("--cv-folds", type=int, default=5, help="交叉驗證折數")
    parser.add_argument("--skip-train", action="store_true", help="跳過訓練，直接使用 model-path 進行推論")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    target_size = (args.image_size, args.image_size)
    if not args.skip_train:
        features, labels, classes = build_dataset(
            args.train_dir,
            target_size,
            args.augmentations_per_image,
        )
        model, metrics, report_text, conf_mat = train_model(
            features,
            labels,
            cross_validation_folds=args.cv_folds,
        )
        print("模型訓練完成。")
        print(f"交叉驗證平均正確率: {metrics['cv_accuracy_mean']:.4f} ± {metrics['cv_accuracy_std']:.4f}")
        print(f"驗證集 Accuracy: {metrics['validation_accuracy']:.4f}")
        print(f"驗證集 Precision 平均: {metrics['validation_precision']:.4f}")
        print(f"驗證集 Recall 平均: {metrics['validation_recall']:.4f}")
        print(f"驗證集 F1 平均: {metrics['validation_f1']:.4f}")
        print("驗證集分類報告:")
        print(report_text)
        print("驗證集混淆矩陣 (列為真實標籤，行為預測標籤):")
        print(conf_mat)
        print(f"標籤順序: {', '.join(classes)}")
        save_model(model, args.model_path)
        print(f"模型已儲存至: {args.model_path}")
    else:
        model, encoder = load_model(args.model_path)
        classes = list(encoder.classes_)
    if args.predict_dir:
        if args.skip_train:
            model, _ = load_model(args.model_path)
        image_paths = list(list_image_files(args.predict_dir))
        if not image_paths:
            print(f"{args.predict_dir} 內未找到影像。")
            return
        counts = count_coins(
            model,
            image_paths,
            target_size,
            label_order=classes,
        )
        for image_name, summary in counts.items():
            readable = ", ".join(f"{label}: {count}" for label, count in summary.items())
            print(f"{image_name}: {readable}")


if __name__ == "__main__":
    main()
