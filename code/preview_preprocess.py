import os
import random
import pandas as pd
import cv2

from preprocess import preprocess_signature, read_image, ensure_dir


def side_by_side(orig_bgr, processed_gray, target_h=300):
    orig_gray = cv2.cvtColor(orig_bgr, cv2.COLOR_BGR2GRAY)

    def resize_to_h(img, h):
        H, W = img.shape[:2]
        new_w = int(W * (h / H))
        return cv2.resize(img, (new_w, h), interpolation=cv2.INTER_AREA)

    o = resize_to_h(orig_gray, target_h)
    p = resize_to_h(processed_gray, target_h)

    sep = 255 * (p[:, :10] * 0 + 1).astype("uint8")
    out = cv2.hconcat([o, sep, p])
    return out


def main():
    base_dir = os.path.dirname(__file__)
    csv_path = os.path.join(base_dir, "data.csv")
    df = pd.read_csv(csv_path)

    sample_n = 30
    rnd = 42
    random.seed(rnd)

    train_g = df[(df["split"] == "train") & (df["label"] == "genuine")]
    test_g = df[(df["split"] == "test") & (df["label"] == "genuine")]
    test_f = df[(df["split"] == "test") & (df["label"] == "forged")]

    parts = []
    parts += train_g.sample(min(10, len(train_g)), random_state=rnd).to_dict("records")
    parts += test_g.sample(min(10, len(test_g)), random_state=rnd).to_dict("records")
    parts += test_f.sample(min(10, len(test_f)), random_state=rnd).to_dict("records")

    if len(parts) < sample_n:
        extra = df.sample(sample_n - len(parts), random_state=rnd).to_dict("records")
        parts += extra

    out_dir = os.path.join(base_dir, "previews")
    ensure_dir(out_dir)

    for i, row in enumerate(parts, start=1):
        img_path = row["image_path"]
        person_id = row["person_id"]
        label = row["label"]
        split = row["split"]

        if not os.path.isabs(img_path):
            img_path = os.path.join(base_dir, img_path)

        orig = read_image(img_path)

        processed, _dbg = preprocess_signature(
            orig,
            out_size=(220, 155),
            binarize_method="otsu",
            invert_if_needed=True,
            morph_open=True,
            open_ksize=2,
            pad=8
        )

        collage = side_by_side(orig, processed, target_h=280)
        fname = f"{i:02d}_id{person_id}_{label}_{split}.png"
        cv2.imwrite(os.path.join(out_dir, fname), collage)

    print(f"Saved previews to: {out_dir}")


if __name__ == "__main__":
    main()
