from pathlib import Path
import sys
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
from torchvision.models import mobilenet_v3_small

INPUT_DIR = Path("inputs")
MODEL_PATH = Path("models/model_finetuned.pth")
IMAGE_SIZE = 64
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def smooth1d(x, w):
    if w <= 1:
        return x
    kernel = np.ones(w, dtype=np.float32) / w
    return np.convolve(x, kernel, mode="same")


def pick_top_peaks(proj, k=3, min_sep=10, min_value=1):
    idxs = []
    arr = proj.copy()
    for _ in range(k):
        i = int(np.argmax(arr))
        if arr[i] < min_value:
            break
        idxs.append(i)
        left = max(0, i - min_sep)
        right = min(len(arr), i + min_sep + 1)
        arr[left:right] = -1.0
    return sorted(idxs)


def expand_interval(proj, peak_idx, edge_frac=0.12, min_width=6):
    peak_val = float(proj[peak_idx])
    if peak_val <= 0:
        return peak_idx, peak_idx + 1
    threshold = peak_val * edge_frac
    left = peak_idx
    while left > 0 and proj[left] > threshold:
        left -= 1
    right = peak_idx
    max_len = len(proj)
    while right < max_len - 1 and proj[right] > threshold:
        right += 1
    if (right - left) < min_width:
        left = max(0, peak_idx - min_width // 2)
        right = min(max_len, left + min_width)
    return left, right


def find_vertical_range(img_bgr, thr=28, dilate=2, pad_y=6):
    hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    _, mask = cv2.threshold(value, thr, 255, cv2.THRESH_BINARY)
    if dilate > 0:
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.dilate(mask, kernel, iterations=dilate)
    ys = np.where(mask.sum(axis=1) > 0)[0]
    if ys.size == 0:
        h = img_bgr.shape[0]
        return 0, max(1, h // 2)
    y1 = int(max(0, ys.min() - pad_y))
    y2 = int(min(img_bgr.shape[0], ys.max() + pad_y + 1))
    return y1, y2


def make_square_and_resize(img_patch, size=64):
    h, w = img_patch.shape[:2]
    side = max(h, w, 8)
    pad_top = (side - h) // 2
    pad_bottom = side - h - pad_top
    pad_left = (side - w) // 2
    pad_right = side - w - pad_left
    padded = cv2.copyMakeBorder(
        img_patch,
        pad_top,
        pad_bottom,
        pad_left,
        pad_right,
        cv2.BORDER_CONSTANT,
        value=(0, 0, 0),
    )
    return cv2.resize(padded, (size, size), interpolation=cv2.INTER_AREA)


def make_contiguous_three_intervals(intervals, total_width):
    if intervals is None or len(intervals) < 3:
        return [
            (0, total_width // 3),
            (total_width // 3, 2 * total_width // 3),
            (2 * total_width // 3, total_width),
        ]

    intervals = sorted(intervals[:3], key=lambda item: item[0])
    (l0, r0), (l1, r1), (l2, r2) = intervals
    b1 = int(round((r0 + l1) / 2.0))
    b2 = int(round((r1 + l2) / 2.0))
    b1 = max(1, min(b1, total_width - 2))
    b2 = max(2, min(b2, total_width - 1))

    if b1 >= b2:
        return [
            (0, total_width // 3),
            (total_width // 3, 2 * total_width // 3),
            (2 * total_width // 3, total_width),
        ]

    return [(0, b1), (b1, b2), (b2, total_width)]


def extract_three_patches(img_bgr, size=64):
    y1, y2 = find_vertical_range(img_bgr, thr=28, dilate=2, pad_y=6)
    top = img_bgr[y1:y2, :]
    _, width = top.shape[:2]

    hsv = cv2.cvtColor(top, cv2.COLOR_BGR2HSV)
    value = hsv[:, :, 2]
    _, mask = cv2.threshold(value, 28, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    mask = cv2.dilate(mask, kernel, iterations=2)

    projection = mask.sum(axis=0).astype(np.float32)
    projection_smooth = smooth1d(projection, 9)

    peaks = pick_top_peaks(projection_smooth, k=3, min_sep=max(6, width // 8), min_value=6)
    if len(peaks) < 3:
        peaks = pick_top_peaks(projection_smooth, k=3, min_sep=max(4, width // 12), min_value=2)

    intervals = []
    for peak in peaks:
        left, right = expand_interval(projection_smooth, peak, edge_frac=0.12, min_width=max(6, width // 20))
        left = max(0, left - 10)
        right = min(width, right + 10)
        intervals.append((left, right))

    intervals = make_contiguous_three_intervals(intervals, width)

    patches = []
    for left, right in intervals:
        crop = top[:, left:right]
        patches.append(make_square_and_resize(crop, size=size))

    return patches


class DigitModel(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_small(weights=None)
        self.features = backbone.features
        with torch.no_grad():
            dummy = torch.zeros(1, 3, IMAGE_SIZE, IMAGE_SIZE)
            feat_dim = self.features(dummy).shape[1]
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.head = nn.Linear(feat_dim, 10)

    def forward(self, x):
        x = self.features(x)
        x = self.pool(x).flatten(1)
        x = self.head(x)
        return x


def load_model():
    model = DigitModel().to(DEVICE)
    state = torch.load(MODEL_PATH, map_location=DEVICE)
    try:
        model.load_state_dict(state)
    except Exception:
        model.load_state_dict(state, strict=False)
    model.eval()
    return model


def predict_image(model, image_path):
    image_bgr = cv2.imread(str(image_path))
    if image_bgr is None:
        raise RuntimeError(f"Failed to load image: {image_path}")

    patches = extract_three_patches(image_bgr, size=IMAGE_SIZE)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    digits = []
    confidences = []

    with torch.no_grad():
        for patch in patches:
            rgb_patch = cv2.cvtColor(patch, cv2.COLOR_BGR2RGB)
            tensor = transform(Image.fromarray(rgb_patch).convert("RGB")).unsqueeze(0).to(DEVICE)
            logits = model(tensor)
            probs = F.softmax(logits, dim=1)[0]
            pred = int(torch.argmax(probs).item())
            conf = float(probs[pred].item())
            digits.append(pred)
            confidences.append(conf)

    result = "".join(str(digit) for digit in digits)
    return result, digits, confidences


def main():
    if not INPUT_DIR.exists():
        print("Missing input folder: inputs/")
        return 1

    if not MODEL_PATH.exists():
        print("Model file not found: models/model_finetuned.pth")
        print("This public repository does not include the trained model weights.")
        print("The model is kept private for safety reasons and to reduce the risk of misuse.")
        return 0

    try:
        model = load_model()
    except Exception as exc:
        print(f"Failed to load model: {exc}")
        return 1

    image_files = sorted(
        path for path in INPUT_DIR.iterdir()
        if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp"}
    )

    if not image_files:
        print("No images found in inputs/")
        return 0

    print("Running predictions...")
    for image_path in image_files:
        try:
            result, digits, confidences = predict_image(model, image_path)
            print(
                f"{image_path.name} -> predicted: {result} | digits: {digits} | "
                f"confidence: {[round(value, 3) for value in confidences]}"
            )
        except Exception as exc:
            print(f"{image_path.name} -> failed: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
