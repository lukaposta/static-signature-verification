# make_csv.py
# Generates a CSV index for the signature dataset.
# Dataset structure:
#   dataset/
#     001/         -> genuine signatures for person 001
#     001_forg/    -> forged signatures for person 001
#     002/
#     002_forg/
# CSV columns:
#   image_path,person_id,label,split
#
# Rules:
#   - For each person: pick N_TRAIN_GENUINE genuine signatures for TRAIN
#   - Remaining genuine signatures -> TEST
#   - All forged signatures -> TEST
#   - Selection is random but reproducible via SEED

from __future__ import annotations

import csv
import random
from pathlib import Path

# -------- CONFIG --------
DATASET_DIR = Path("dataset55")  # folder name inside your project
OUTPUT_CSV = Path("data.csv")
N_TRAIN_GENUINE = 18
SEED = 42

ALLOWED_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff"}
# ------------------------


def is_forg_dir(dir_name: str) -> bool:
    return dir_name.endswith("_forg")


def person_id_from_dir(dir_name: str) -> str:
    # "001" -> "001", "001_forg" -> "001"
    return dir_name.replace("_forg", "")


def list_images(folder: Path) -> list[Path]:
    files: list[Path] = []
    for p in folder.iterdir():
        if p.is_file() and p.suffix.lower() in ALLOWED_EXTS:
            files.append(p)
    # stable order before random sampling (important for reproducibility)
    files.sort(key=lambda x: x.name)
    return files


def main() -> None:
    if not DATASET_DIR.exists() or not DATASET_DIR.is_dir():
        raise SystemExit(
            f"ERROR: Can't find dataset folder at '{DATASET_DIR.resolve()}'.\n"
            "Open VS Code in the folder that CONTAINS the 'dataset' folder, then run again."
        )

    rng = random.Random(SEED)

    # Build map: person_id -> {'genuine': [...], 'forged': [...]}
    persons: dict[str, dict[str, list[Path]]] = {}

    for sub in sorted(DATASET_DIR.iterdir(), key=lambda p: p.name):
        if not sub.is_dir():
            continue

        pid = person_id_from_dir(sub.name)
        kind = "forged" if is_forg_dir(sub.name) else "genuine"

        imgs = list_images(sub)
        if not imgs:
            continue

        persons.setdefault(pid, {"genuine": [], "forged": []})
        persons[pid][kind].extend(imgs)

    if not persons:
        raise SystemExit("ERROR: No persons found. Check dataset folder structure and file extensions.")

    rows: list[dict[str, str]] = []

    dropped_persons: list[str] = []
    for pid, data in sorted(persons.items(), key=lambda x: x[0]):
        genuine = sorted(data["genuine"], key=lambda p: p.name)
        forged = sorted(data["forged"], key=lambda p: p.name)

        if len(genuine) < N_TRAIN_GENUINE:
            dropped_persons.append(pid)
            continue

        train_genuine = set(rng.sample(genuine, N_TRAIN_GENUINE))
        test_genuine = [p for p in genuine if p not in train_genuine]

        # Add genuine rows
        for p in genuine:
            split = "train" if p in train_genuine else "test"
            rows.append(
                {
                    "image_path": p.as_posix(),
                    "person_id": pid,
                    "label": "genuine",
                    "split": split,
                }
            )

        # Add forged rows (all test)
        for p in forged:
            rows.append(
                {
                    "image_path": p.as_posix(),
                    "person_id": pid,
                    "label": "forged",
                    "split": "test",
                }
            )

    # Write CSV
    OUTPUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["image_path", "person_id", "label", "split"])
        writer.writeheader()
        writer.writerows(rows)

    # Summary
    n_train = sum(1 for r in rows if r["split"] == "train")
    n_test = sum(1 for r in rows if r["split"] == "test")
    n_genuine = sum(1 for r in rows if r["label"] == "genuine")
    n_forged = sum(1 for r in rows if r["label"] == "forged")

    print("CSV generated:", OUTPUT_CSV.resolve())
    print("Persons included:", len(persons) - len(dropped_persons))
    print("Persons dropped (not enough genuine):", len(dropped_persons))
    if dropped_persons:
        print("Dropped IDs (first 20):", ", ".join(dropped_persons[:20]))
    print("Total images:", len(rows))
    print("Genuine:", n_genuine, "| Forged:", n_forged)
    print("Train:", n_train, "| Test:", n_test)
    print(f"Train genuine per person: {N_TRAIN_GENUINE} (seed={SEED})")


if __name__ == "__main__":
    main()
