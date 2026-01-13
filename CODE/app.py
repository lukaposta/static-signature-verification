import os
import math
from pathlib import Path

import cv2
import joblib
import numpy as np
from flask import Flask, render_template, request, send_from_directory
from werkzeug.utils import secure_filename
from skimage.feature import hog

from preprocess import preprocess_signature, read_image

APP_DIR = Path(__file__).resolve().parent
MODELS_DIR = APP_DIR / "models"
UPLOADS_DIR = APP_DIR / "uploads"

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
MAX_PERSONS_IN_DROPDOWN = 55

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB


def list_person_ids(models_dir: Path) -> list[str]:
    if not models_dir.exists():
        return []
    ids = []
    for p in sorted(models_dir.glob("svm_person_*.joblib")):
        name = p.stem
        pid = name.replace("svm_person_", "")
        ids.append(pid)
    return ids[:MAX_PERSONS_IN_DROPDOWN]


def extract_hog(gray_img: np.ndarray, hog_cfg: dict) -> np.ndarray:
    feat = hog(
        gray_img,
        orientations=hog_cfg["orientations"],
        pixels_per_cell=tuple(hog_cfg["pixels_per_cell"]),
        cells_per_block=tuple(hog_cfg["cells_per_block"]),
        block_norm=hog_cfg["block_norm"],
        transform_sqrt=hog_cfg["transform_sqrt"],
        feature_vector=True,
    )
    return feat.astype(np.float32)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def validate_upload(filename: str) -> bool:
    ext = Path(filename).suffix.lower()
    return ext in ALLOWED_EXTS


def load_model_pack(person_id: str) -> dict:
    model_path = MODELS_DIR / f"svm_person_{person_id}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Model ne postoji: {model_path}")
    return joblib.load(model_path)


def run_verify(person_id: str, image_path: Path) -> dict:
    pack = load_model_pack(person_id)

    clf = pack["model"]
    prep_cfg = pack["prep_cfg"]
    hog_cfg = pack["hog_cfg"]
    thr = float(pack.get("threshold", 0.0))

    bgr = read_image(str(image_path))
    proc, _dbg = preprocess_signature(
        bgr,
        out_size=tuple(prep_cfg["out_size"]),
        binarize_method=prep_cfg["binarize_method"],
        invert_if_needed=prep_cfg["invert_if_needed"],
        morph_open=prep_cfg["morph_open"],
        open_ksize=prep_cfg["open_ksize"],
        pad=prep_cfg["pad"],
    )

    x = extract_hog(proc, hog_cfg).reshape(1, -1)

    score = float(clf.decision_function(x)[0])
    accept = score >= thr

    p = sigmoid(score)
    if accept:
        decision = "ACCEPT"
        confidence = p * 100.0
        conf_label = "Confidence in ACCEPT"
    else:
        decision = "REJECT"
        confidence = (1.0 - p) * 100.0
        conf_label = "Confidence in REJECT"

    return {
        "decision": decision,
        "confidence": confidence,
        "confidence_label": conf_label,
        "score": score,
        "threshold": thr,
        "proc_img": proc,
    }


@app.route("/uploads/<path:filename>")
def uploads(filename):
    return send_from_directory(str(UPLOADS_DIR), filename)


@app.route("/", methods=["GET", "POST"])
def index():
    persons = list_person_ids(MODELS_DIR)

    result = None
    error = None
    selected_person = persons[0] if persons else ""

    original_url = None
    processed_url = None

    if request.method == "POST":
        selected_person = request.form.get("person_id", "").strip()
        f = request.files.get("image")

        try:
            if not persons:
                raise RuntimeError("Nema modela u code/models (svm_person_{id}.joblib).")

            if selected_person == "" or selected_person not in persons:
                raise ValueError("Neispravan odabir osobe.")

            if f is None or f.filename is None or f.filename.strip() == "":
                raise ValueError("Nisi odabrao sliku za upload.")

            if not validate_upload(f.filename):
                raise ValueError("Nepodržan format. Dozvoljeno: png, jpg, jpeg, tif, tiff.")

            UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
            safe_name = secure_filename(f.filename)
            save_path = UPLOADS_DIR / safe_name
            f.save(str(save_path))

            out = run_verify(selected_person, save_path)

            # spremi preprocessed sliku
            proc = out.pop("proc_img")
            proc_name = f"{Path(safe_name).stem}_proc.png"
            proc_path = UPLOADS_DIR / proc_name
            cv2.imwrite(str(proc_path), proc)

            # URL-ovi za prikaz u HTML-u
            original_url = f"/uploads/{safe_name}"
            processed_url = f"/uploads/{proc_name}"

            result = out

        except Exception as e:
            error = str(e)

    return render_template(
        "index.html",
        persons=persons,
        selected_person=selected_person,
        result=result,
        error=error,
        original_url=original_url,
        processed_url=processed_url,
    )


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
