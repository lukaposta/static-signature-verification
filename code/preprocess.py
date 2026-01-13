import os
import cv2
import numpy as np


def preprocess_signature(
    img_bgr: np.ndarray,
    out_size=(220, 155),
    binarize_method="otsu",
    invert_if_needed=True,
    morph_open=True,
    open_ksize=2,
    pad=8,
):
    """
    Pipeline:
    1) grayscale
    2) binarization (otsu/adaptive)
    3) (optional) invert so ink=white on black for bbox logic
    4) (optional) small morphology open to remove specks
    5) crop to bounding box
    6) resize to fixed size

    Returns:
      processed: uint8 image in [0,255], shape = (H,W) = out_size reversed in cv2 resize usage
      debug: dict with intermediate images (for preview/debug)
    """

    debug = {}

    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    debug["gray"] = gray

    if binarize_method == "adaptive":
        # Adaptive threshold works better when background illumination varies
        bin_img = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 5
        )
    else:
        # Otsu is usually enough for scanned signatures
        _, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    debug["bin"] = bin_img

    # Decide foreground vs background (ink should be 255 for bbox detection convenience)
    # If background is mostly white, threshold result often has background=255, ink=0
    # We flip so ink becomes 255.
    ink_white = bin_img.copy()
    if invert_if_needed:
        # If there are more white pixels than black pixels, likely background is white.
        # In that case ink is black -> invert.
        white_ratio = (bin_img == 255).mean()
        if white_ratio > 0.5:
            ink_white = 255 - bin_img

    debug["ink_white"] = ink_white

    # Remove tiny noise, keep strokes
    mask = ink_white
    if morph_open:
        k = max(1, int(open_ksize))
        kernel = np.ones((k, k), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)

    debug["mask"] = mask

    # Bounding box on non-zero pixels
    ys, xs = np.where(mask > 0)
    if len(xs) == 0 or len(ys) == 0:
        # Fallback: empty/failed image -> return resized grayscale to keep pipeline alive
        resized = cv2.resize(gray, out_size, interpolation=cv2.INTER_AREA)
        debug["crop"] = resized
        debug["final"] = resized
        return resized, debug

    x0, x1 = xs.min(), xs.max()
    y0, y1 = ys.min(), ys.max()

    # pad bbox a bit so strokes do not touch edges
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(gray.shape[1] - 1, x1 + pad)
    y1 = min(gray.shape[0] - 1, y1 + pad)

    crop = gray[y0:y1 + 1, x0:x1 + 1]
    debug["crop"] = crop

    # Resize
    final_img = cv2.resize(crop, out_size, interpolation=cv2.INTER_AREA)
    debug["final"] = final_img

    return final_img, debug


def read_image(path: str):
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Cannot read image: {path}")
    return img


def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)
