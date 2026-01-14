import argparse
import random
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple, Union, Iterator

import cv2
import joblib
import numpy as np
from skimage.feature import hog, local_binary_pattern
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import LinearSVC, SVC

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

RANDOM_SEED = 42
LBP_POINTS = 24
LBP_RADIUS = 3
LBP_BINS = LBP_POINTS + 2

# [參數] 裁切推論時稍微放大
CROP_SCALE_INFERENCE = 1.2

# [V8 參數] 雙重驗證門檻
# 1. 紋理密度閾值 (3.5%): 必須有足夠的線條細節
TEXTURE_DENSITY_THRESHOLD = 0.035
# 2. 標準差閾值 (20.0): 必須有足夠的對比度 (金屬反光)
# 普通桌面/陰影通常 < 15.0，硬幣通常 > 30.0
STD_THRESHOLD = 20.0

def to_numpy(array) -> np.ndarray:
    if hasattr(array, "get"):
        return np.asarray(array.get())
    return np.asarray(array)

class CuMLPredictWrapper:
    def __init__(self, estimator):
        self.estimator = estimator
    def fit(self, X, y):
        self.estimator.fit(X, y)
        return self
    def predict(self, X):
        preds = self.estimator.predict(X)
        return to_numpy(preds)
    def score(self, X, y):
        preds = self.predict(X)
        return float(np.mean(preds == y))
    def get_params(self, deep=True):
        return {"estimator": self.estimator}
    def set_params(self, **params):
        if "estimator" in params:
            self.estimator = params["estimator"]
        return self

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

# === Augmentation Utilities ===
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
    shift_x, shift_y = max_shift_ratio * w, max_shift_ratio * h
    src = np.float32([[0, 0], [w - 1, 0], [0, h - 1], [w - 1, h - 1]])
    dst = np.float32([
        [random.uniform(0, shift_x), random.uniform(0, shift_y)],
        [w - 1 - random.uniform(0, shift_x), random.uniform(0, shift_y)],
        [random.uniform(0, shift_x), h - 1 - random.uniform(0, shift_y)],
        [w - 1 - random.uniform(0, shift_x), h - 1 - random.uniform(0, shift_y)],
    ])
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
    return (image.astype(np.float32) * (1 - alpha[..., None])).astype(np.uint8)

def augment_coin_generator(image: np.ndarray, count: int) -> Iterator[np.ndarray]:
    for _ in range(count):
        variant = image.copy()
        if random.random() < 0.6: variant = rotate_image(variant, random.uniform(-28, 28))
        if random.random() < 0.5: variant = scale_image(variant, random.uniform(0.9, 1.12))
        if random.random() < 0.45:
            variant = translate_image(variant, 
                random.randint(-int(0.06 * variant.shape[1]), int(0.06 * variant.shape[1])), 
                random.randint(-int(0.06 * variant.shape[0]), int(0.06 * variant.shape[0])))
        if random.random() < 0.3: variant = perspective_transform(variant, max_shift_ratio=0.06)
        if random.random() < 0.5: variant = adjust_gamma(variant, random.uniform(0.75, 1.35))
        if random.random() < 0.3: variant = add_gaussian_noise(variant, std=random.uniform(4, 12))
        if random.random() < 0.2: variant = random_shadow(variant, opacity=random.uniform(0.18, 0.4))
        yield variant

# === Feature Extraction (Color + HOG + LBP) ===
def preprocess_image(image: np.ndarray, target_size: Tuple[int, int]) -> np.ndarray:
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)

def extract_features(image_bgr: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    mean, std = cv2.meanStdDev(hsv)
    color_feats = np.concatenate([mean, std]).flatten()
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
    hog_features = hog(
        gray, orientations=9, pixels_per_cell=(16, 16),
        cells_per_block=(2, 2), block_norm="L2-Hys", transform_sqrt=True, feature_vector=True,
    )
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    hist, _ = np.histogram(lbp, bins=LBP_BINS, range=(0, LBP_BINS))
    hist = hist.astype(np.float32)
    hist /= (hist.sum() + 1e-6)
    return np.concatenate([color_feats, hog_features, hist])

def build_dataset(train_dir: Path, target_size: Tuple[int, int], augmentations_per_image: int) -> Tuple[np.ndarray, np.ndarray, List[str]]:
    random.seed(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)
    images, labels = [], []
    if not train_dir.exists(): raise ValueError(f"找不到路徑: {train_dir}")
    classes = sorted([d.name for d in train_dir.iterdir() if d.is_dir()])
    if not classes: raise ValueError(f"在 {train_dir} 找不到分類資料夾")
    processing_limit = max(target_size[0], target_size[1]) * 3
    for class_name in classes:
        class_dir = train_dir / class_name
        image_list = list(list_image_files(class_dir))
        for image_path in tqdm(image_list, desc=f"載入 {class_name}", unit="img"):
            image = cv2.imread(str(image_path))
            if image is None: continue
            h, w = image.shape[:2]
            if h > processing_limit or w > processing_limit:
                scale = min(processing_limit / h, processing_limit / w)
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            base = ensure_color(image)
            resized_base = preprocess_image(base, target_size)
            images.append(extract_features(resized_base))
            labels.append(class_name)
            aug_gen = augment_coin_generator(base, augmentations_per_image)
            for aug in aug_gen:
                resized_aug = preprocess_image(aug, target_size)
                images.append(extract_features(resized_aug))
                labels.append(class_name)
    return np.asarray(images), np.asarray(labels), classes

def train_model(features: np.ndarray, labels: np.ndarray, cross_validation_folds: int, svm_kernel: str, use_gpu: bool) -> Tuple[Pipeline, LabelEncoder, Dict[str, float], str, np.ndarray]:
    encoder = LabelEncoder()
    encoded_labels = encoder.fit_transform(labels)
    estimator = None
    if use_gpu:
        try:
            from cuml.svm import LinearSVC as CuMLLinearSVC, SVC as CuMLSVC
            estimator = CuMLLinearSVC(class_weight="balanced") if svm_kernel == "linear" else CuMLSVC(kernel="rbf", class_weight="balanced")
        except ImportError: print("⚠️ 啟用 --use-gpu 但未安裝 cuML，改用 CPU 版 SVM。")
    if estimator is None:
        estimator = LinearSVC(class_weight="balanced", random_state=RANDOM_SEED) if svm_kernel == "linear" else SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_SEED)
    elif use_gpu: estimator = CuMLPredictWrapper(estimator)
    pipeline = Pipeline([("scaler", StandardScaler(with_mean=False)), ("svm", estimator)])
    x_train, x_valid, y_train, y_valid = train_test_split(features, encoded_labels, test_size=0.2, random_state=RANDOM_SEED, stratify=encoded_labels)
    pipeline.fit(x_train, y_train)
    y_valid_pred = to_numpy(pipeline.predict(x_valid))
    report_dict = classification_report(encoder.inverse_transform(y_valid), encoder.inverse_transform(y_valid_pred), labels=encoder.classes_, target_names=encoder.classes_, zero_division=0, output_dict=True)
    report_text = classification_report(encoder.inverse_transform(y_valid), encoder.inverse_transform(y_valid_pred), labels=encoder.classes_, target_names=encoder.classes_, digits=4, zero_division=0)
    conf_mat = confusion_matrix(encoder.inverse_transform(y_valid), encoder.inverse_transform(y_valid_pred), labels=encoder.classes_)
    cv_scores = []
    skf = StratifiedKFold(n_splits=cross_validation_folds, shuffle=True, random_state=RANDOM_SEED)
    for train_idx, test_idx in tqdm(skf.split(features, encoded_labels), total=cross_validation_folds, desc="交叉驗證", unit="fold"):
        pipeline.fit(features[train_idx], encoded_labels[train_idx])
        cv_scores.append(pipeline.score(features[test_idx], encoded_labels[test_idx]))
    pipeline.fit(features, encoded_labels)
    metrics = {
        "validation_accuracy": float(report_dict["accuracy"]),
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_std": float(np.std(cv_scores)),
    }
    return pipeline, encoder, metrics, report_text, conf_mat

# === 核心檢測邏輯 (V8 雙重驗證版) ===

def dedup_circles(circles: Sequence[Tuple[int, int, int]], image_shape: Tuple[int, int, int]) -> List[Tuple[int, int, int]]:
    """智慧去重"""
    if not circles: return []
    kept = []
    for cx, cy, r in circles:
        is_dup = False
        for kx, ky, kr in kept:
            dist_sq = (cx - kx) ** 2 + (cy - ky) ** 2
            # [修正] 激進去重：距離小於較大半徑的 100% 就視為重疊
            # 這能強迫合併非常靠近的兩個圓
            overlap_thresh = (max(r, kr) * 1.0) ** 2
            if dist_sq < overlap_thresh:
                is_dup = True
                break
        if not is_dup:
            kept.append((cx, cy, r))
    return kept

def is_valid_coin_crop(crop: np.ndarray) -> bool:
    """[V8] 雙重驗證：同時檢查 標準差(對比度) 與 紋理密度"""
    if crop.size == 0: return False
    
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    
    # 1. 對比度檢查 (標準差)
    mean, std = cv2.meanStdDev(gray)
    if std[0][0] < STD_THRESHOLD: # < 20.0 直接踢掉
        return False
        
    # 2. 紋理密度檢查
    edges = cv2.Canny(gray, 50, 150)
    h, w = gray.shape
    mask = np.zeros((h, w), dtype=np.uint8)
    # 只看中心 60%，完全避開邊緣陰影
    cv2.circle(mask, (w//2, h//2), int(min(h, w)/2 * 0.6), 255, -1)
    
    masked_edges = cv2.bitwise_and(edges, edges, mask=mask)
    edge_pixels = cv2.countNonZero(masked_edges)
    total_pixels = cv2.countNonZero(mask)
    
    density = edge_pixels / (total_pixels + 1e-6)
    
    # 密度 < 3.5% 直接踢掉
    return density > TEXTURE_DENSITY_THRESHOLD

def detect_coins(
    image: np.ndarray,
    dp: float = 1.2,
    min_dist_ratio: float = 0.04,
    param1: float = 150,
    param2: float = 38,             # [修正] 拉高到 38，這是非常嚴格的數值，只讓很圓的東西通過
    min_radius_ratio: float = 0.02, 
    max_radius_ratio: float = 0.16, 
    use_clahe: bool = True, 
) -> List[Tuple[int, int, int]]:
    
    bgr = ensure_color(image)
    if use_clahe:
        lab = cv2.cvtColor(bgr, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        lab = cv2.merge((l, a, b))
        bgr = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    
    min_dim = min(image.shape[:2])
    min_dist = max(15, int(min_dim * min_dist_ratio))
    min_radius = int(min_dim * min_radius_ratio)
    max_radius = int(min_dim * max_radius_ratio)
    
    circles = cv2.HoughCircles(
        gray, cv2.HOUGH_GRADIENT, dp=dp, minDist=min_dist,
        param1=param1, param2=param2, minRadius=min_radius, maxRadius=max_radius,
    )
    
    all_circles = []
    if circles is not None:
        all_circles = [(int(x), int(y), int(r)) for x, y, r in circles[0]]
            
    return all_circles

def crop_coin(image: np.ndarray, circle: Tuple[int, int, int], scale: float = 1.0) -> np.ndarray:
    x, y, r = circle
    r_scaled = int(r * scale)
    x1 = max(x - r_scaled, 0)
    y1 = max(y - r_scaled, 0)
    x2 = min(x + r_scaled, image.shape[1])
    y2 = min(y + r_scaled, image.shape[0])
    return image[y1:y2, x1:x2]

def count_coins(model: Pipeline, image_paths: Sequence[Path], target_size: Tuple[int, int], label_order: Sequence[str]) -> Dict[str, Dict[str, int]]:
    encoder = getattr(model, "label_encoder", None) or model.named_steps.get("label_encoder")
    counts = {}
    for path in image_paths:
        image = cv2.imread(str(path))
        if image is None: continue
        image = ensure_color(image)
        circles = detect_coins(image)
        circles = dedup_circles(circles, image.shape)
        coin_counter = Counter()
        for circle in circles:
            crop = crop_coin(image, circle, scale=CROP_SCALE_INFERENCE)
            if not is_valid_coin_crop(crop):
                continue
            resized_crop = preprocess_image(crop, target_size)
            features = extract_features(resized_crop)
            prediction = to_numpy(model.predict([features]))[0]
            label = encoder.inverse_transform([prediction])[0]
            coin_counter[label] += 1
        counts[path.name] = {label: coin_counter.get(label, 0) for label in label_order}
    return counts

def save_model(model: Pipeline, encoder: LabelEncoder, destination: Path) -> None:
    model_copy = Pipeline(model.steps[:-1] + [("svm", model.named_steps["svm"])])
    payload = {"model": model_copy, "encoder": encoder}
    joblib.dump(payload, destination)

def load_model(path: Path) -> Tuple[Pipeline, LabelEncoder]:
    payload = joblib.load(path)
    model = payload["model"]
    encoder = payload["encoder"]
    model.named_steps["label_encoder"] = encoder
    model.label_encoder = encoder
    return model, encoder

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="硬幣分類與計數 (Color + HOG + LBP + SVM)")
    parser.add_argument("--train-dir", type=Path, default=Path("train_coins"), help="訓練資料夾")
    parser.add_argument("--predict-dir", type=Path, help="推論影像資料夾")
    parser.add_argument("--model-path", type=Path, default=Path("coin_svm_v7.joblib"), help="模型路徑")
    parser.add_argument("--image-size", type=int, default=112, help="影像尺寸")
    parser.add_argument("--augmentations-per-image", type=int, default=10, help="擴增數量")
    parser.add_argument("--cv-folds", type=int, default=3, help="交叉驗證折數")
    parser.add_argument("--svm-kernel", type=str, choices=["rbf", "linear"], default="rbf", help="SVM kernel")
    parser.add_argument("--skip-train", action="store_true", help="跳過訓練")
    parser.add_argument("--use-gpu", action="store_true", help="啟用 GPU")
    return parser.parse_args()

def main() -> None:
    args = parse_args()
    target_size = (args.image_size, args.image_size)
    if not args.skip_train:
        features, labels, classes = build_dataset(args.train_dir, target_size, args.augmentations_per_image)
        model, encoder, metrics, report_text, conf_mat = train_model(features, labels, args.cv_folds, args.svm_kernel, args.use_gpu)
        print("模型訓練完成。")
        print(f"交叉驗證平均正確率: {metrics['cv_accuracy_mean']:.4f}")
        print(report_text)
        model.named_steps["label_encoder"] = encoder
        model.label_encoder = encoder
        save_model(model, encoder, args.model_path)
    else:
        model, encoder = load_model(args.model_path)
        classes = list(encoder.classes_)
    if args.predict_dir:
        if args.skip_train: model, _ = load_model(args.model_path)
        image_paths = list(list_image_files(args.predict_dir))
        counts = count_coins(model, image_paths, target_size, label_order=classes)
        for name, summary in counts.items():
            readable = ", ".join(f"{label}: {count}" for label, count in summary.items())
            print(f"{name}: {readable}")

if __name__ == "__main__":
    main()