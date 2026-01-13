import os
import argparse
import joblib
import numpy as np
from skimage.feature import hog

from preprocess import preprocess_signature, read_image


def extract_hog(gray_img: np.ndarray, hog_cfg: dict) -> np.ndarray:
    feat = hog(
        gray_img,
        orientations=hog_cfg["orientations"],
        pixels_per_cell=hog_cfg["pixels_per_cell"],
        cells_per_block=hog_cfg["cells_per_block"],
        block_norm=hog_cfg["block_norm"],
        transform_sqrt=hog_cfg["transform_sqrt"],
        feature_vector=True,
    )
    return feat.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--person_id", required=True)
    ap.add_argument("--image", required=True)
    ap.add_argument("--models_dir", default="models")
    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    model_path = os.path.join(base_dir, args.models_dir, f"svm_person_{args.person_id}.joblib")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model ne postoji: {model_path}")

    pack = joblib.load(model_path)
    clf = pack["model"]
    prep_cfg = pack["prep_cfg"]
    hog_cfg = pack["hog_cfg"]
    thr = float(pack.get("threshold", 0.0))

    img_path = args.image
    if not os.path.isabs(img_path):
        img_path = os.path.join(base_dir, img_path)

    bgr = read_image(img_path)
    proc, _dbg = preprocess_signature(
        bgr,
        out_size=prep_cfg["out_size"],
        binarize_method=prep_cfg["binarize_method"],
        invert_if_needed=prep_cfg["invert_if_needed"],
        morph_open=prep_cfg["morph_open"],
        open_ksize=prep_cfg["open_ksize"],
        pad=prep_cfg["pad"],
    )

    x = extract_hog(proc, hog_cfg).reshape(1, -1)

    # SVC u pipelineu ima decision_function
    score = float(clf.decision_function(x)[0])
    pred = 1 if score >= thr else 0

    if pred == 1:
        print("ACCEPT")
    else:
        print("REJECT")
    print("score:", round(score, 6))
    print("threshold:", thr)


if __name__ == "__main__":
    main()


# HOW TO RUN:

#GENUINE:
#python verify_one.py --person_id 120 --image dataset55/120/original_1_14.png


#FORGED:
#python verify_one.py --person_id 129 --image dataset55/143_forg/forgeries_24_6.png