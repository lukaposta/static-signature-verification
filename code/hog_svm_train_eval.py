import os
import argparse
import numpy as np
import pandas as pd
import cv2
from skimage.feature import hog
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, accuracy_score
import joblib

from preprocess import preprocess_signature, read_image, ensure_dir


def resolve_path(base_dir: str, p: str) -> str:
    if os.path.isabs(p):
        return p
    return os.path.join(base_dir, p)


def extract_hog(gray_img: np.ndarray, hog_cfg: dict) -> np.ndarray:
    # gray_img expected uint8, shape (H, W)
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


def build_feature_cache(df: pd.DataFrame, base_dir: str, prep_cfg: dict, hog_cfg: dict, cache_path: str):
    ensure_dir(os.path.dirname(cache_path))

    feats = np.zeros((len(df), 1), dtype=np.float32)
    ok = np.zeros((len(df),), dtype=np.uint8)

    feature_list = []
    for idx, row in enumerate(df.itertuples(index=False)):
        img_path = resolve_path(base_dir, getattr(row, "image_path"))
        try:
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
            feature_list.append(extract_hog(proc, hog_cfg))
            ok[idx] = 1
        except Exception:
            feature_list.append(None)
            ok[idx] = 0

    # Filter failures
    valid_idx = np.where(ok == 1)[0]
    if len(valid_idx) == 0:
        raise RuntimeError("Nema niti jedne uspješne ekstrakcije značajki. Provjeri putanje i preprocess.")

    X = np.stack([feature_list[i] for i in valid_idx], axis=0)
    df_valid = df.iloc[valid_idx].reset_index(drop=True)

    np.savez_compressed(
        cache_path,
        X=X,
        image_path=df_valid["image_path"].to_numpy(),
        person_id=df_valid["person_id"].to_numpy(),
        label=df_valid["label"].to_numpy(),
        split=df_valid["split"].to_numpy(),
        prep_cfg=np.array([prep_cfg], dtype=object),
        hog_cfg=np.array([hog_cfg], dtype=object),
    )
    return cache_path


def load_cache(cache_path: str):
    data = np.load(cache_path, allow_pickle=True)
    X = data["X"]
    df = pd.DataFrame({
        "image_path": data["image_path"],
        "person_id": data["person_id"],
        "label": data["label"],
        "split": data["split"],
    })
    prep_cfg = data["prep_cfg"][0]
    hog_cfg = data["hog_cfg"][0]
    return X, df, prep_cfg, hog_cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default="data.csv")
    ap.add_argument("--dataset_base", default=".")
    ap.add_argument("--cache", default="cache/features_hog.npz")
    ap.add_argument("--rebuild_cache", action="store_true")

    ap.add_argument("--out_models", default="models")

    # Preprocess cfg
    ap.add_argument("--out_w", type=int, default=220)
    ap.add_argument("--out_h", type=int, default=155)
    ap.add_argument("--binarize", default="otsu", choices=["otsu", "adaptive"])
    ap.add_argument("--invert", action="store_true")
    ap.add_argument("--no_invert", action="store_true")
    ap.add_argument("--morph_open", action="store_true")
    ap.add_argument("--no_morph_open", action="store_true")
    ap.add_argument("--open_ksize", type=int, default=2)
    ap.add_argument("--pad", type=int, default=8)

    # HOG cfg
    ap.add_argument("--orientations", type=int, default=9)
    ap.add_argument("--ppc", type=int, default=8)
    ap.add_argument("--cpb", type=int, default=2)
    ap.add_argument("--block_norm", default="L2-Hys")
    ap.add_argument("--transform_sqrt", action="store_true")

    # Training
    ap.add_argument("--neg_ratio", type=float, default=5.0)  # negatives per positive
    ap.add_argument("--kernel", default="rbf", choices=["linear", "rbf"])
    ap.add_argument("--C", type=float, default=1.0)
    ap.add_argument("--gamma", default="scale")
    ap.add_argument("--seed", type=int, default=42)

    args = ap.parse_args()

    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, args.csv)
    dataset_base = os.path.join(base_dir, args.dataset_base)

    df = pd.read_csv(csv_path)

    prep_cfg = {
        "out_size": (args.out_w, args.out_h),
        "binarize_method": args.binarize,
        "invert_if_needed": True,
        "morph_open": True,
        "open_ksize": args.open_ksize,
        "pad": args.pad,
    }
    if args.no_invert:
        prep_cfg["invert_if_needed"] = False
    if args.invert:
        prep_cfg["invert_if_needed"] = True
    if args.no_morph_open:
        prep_cfg["morph_open"] = False
    if args.morph_open:
        prep_cfg["morph_open"] = True

    hog_cfg = {
        "orientations": args.orientations,
        "pixels_per_cell": (args.ppc, args.ppc),
        "cells_per_block": (args.cpb, args.cpb),
        "block_norm": args.block_norm,
        "transform_sqrt": bool(args.transform_sqrt),
    }

    cache_path = os.path.join(base_dir, args.cache)

    if args.rebuild_cache or (not os.path.exists(cache_path)):
        print("Building HOG cache...")
        build_feature_cache(df, dataset_base, prep_cfg, hog_cfg, cache_path)

    X, dfX, prep_cfg_loaded, hog_cfg_loaded = load_cache(cache_path)

    # Split subsets
    train_g = dfX[(dfX["split"] == "train") & (dfX["label"] == "genuine")].copy()
    test_g = dfX[(dfX["split"] == "test") & (dfX["label"] == "genuine")].copy()
    test_f = dfX[(dfX["split"] == "test") & (dfX["label"] == "forged")].copy()

    # Map row index in dfX to feature row index in X
    # dfX is aligned with X row-for-row after cache load
    train_g_idx = train_g.index.to_numpy()
    test_g_idx = test_g.index.to_numpy()
    test_f_idx = test_f.index.to_numpy()

    rng = np.random.default_rng(args.seed)

    persons = np.sort(train_g["person_id"].unique())

    ensure_dir(os.path.join(base_dir, args.out_models))

    per_person = []
    y_all = []
    yhat_all = []

    print(f"Persons (train genuine): {len(persons)}")

    for pid in persons:
        pos_train_mask = (train_g["person_id"] == pid).to_numpy()
        pos_train_idx = train_g_idx[pos_train_mask]
        if len(pos_train_idx) < 2:
            continue

        # Negatives: genuine train from other persons
        neg_pool_mask = (train_g["person_id"] != pid).to_numpy()
        neg_pool_idx = train_g_idx[neg_pool_mask]

        n_pos = len(pos_train_idx)
        n_neg = int(np.round(args.neg_ratio * n_pos))
        if n_neg > len(neg_pool_idx):
            n_neg = len(neg_pool_idx)

        neg_train_idx = rng.choice(neg_pool_idx, size=n_neg, replace=False)

        X_pos = X[pos_train_idx]
        X_neg = X[neg_train_idx]

        X_train = np.vstack([X_pos, X_neg])
        y_train = np.hstack([np.ones((len(X_pos),), dtype=int), np.zeros((len(X_neg),), dtype=int)])

        # Test for this person
        pos_test_idx = test_g_idx[(test_g["person_id"] == pid).to_numpy()]
        neg_test_idx = test_f_idx[(test_f["person_id"] == pid).to_numpy()]

        if len(pos_test_idx) == 0 or len(neg_test_idx) == 0:
            continue

        X_test = np.vstack([X[pos_test_idx], X[neg_test_idx]])
        y_test = np.hstack([np.ones((len(pos_test_idx),), dtype=int), np.zeros((len(neg_test_idx),), dtype=int)])

        clf = Pipeline(steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            ("svm", SVC(
                kernel=args.kernel,
                C=args.C,
                gamma=args.gamma,
                class_weight="balanced",
                probability=False,
            ))
        ])

        clf.fit(X_train, y_train)
        y_hat = clf.predict(X_test)

        tn, fp, fn, tp = confusion_matrix(y_test, y_hat, labels=[0, 1]).ravel()

        far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
        frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        acc = accuracy_score(y_test, y_hat)

        per_person.append({
            "person_id": pid,
            "n_train_pos": int(n_pos),
            "n_train_neg": int(n_neg),
            "n_test_pos": int(len(pos_test_idx)),
            "n_test_neg_forged": int(len(neg_test_idx)),
            "FAR": float(far),
            "FRR": float(frr),
            "ACC": float(acc),
            "TN": int(tn), "FP": int(fp), "FN": int(fn), "TP": int(tp),
        })

        y_all.append(y_test)
        yhat_all.append(y_hat)

        # Save model per person
        model_pack = {
            "person_id": pid,
            "model": clf,
            "prep_cfg": prep_cfg_loaded,
            "hog_cfg": hog_cfg_loaded,
            "threshold": 0.0,  # decision_function threshold for accept
        }
        out_path = os.path.join(base_dir, args.out_models, f"svm_person_{pid}.joblib")
        joblib.dump(model_pack, out_path)

    if len(per_person) == 0:
        raise RuntimeError("Nema niti jedne osobe s kompletnim test setom (genuine test + forged test).")

    res = pd.DataFrame(per_person).sort_values("person_id").reset_index(drop=True)

    # Macro averages
    macro_far = res["FAR"].mean()
    macro_frr = res["FRR"].mean()
    macro_acc = res["ACC"].mean()

    # Global (micro) across all persons
    y_all = np.concatenate(y_all, axis=0)
    yhat_all = np.concatenate(yhat_all, axis=0)
    tn, fp, fn, tp = confusion_matrix(y_all, yhat_all, labels=[0, 1]).ravel()

    micro_far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    micro_frr = fn / (fn + tp) if (fn + tp) > 0 else 0.0
    micro_acc = accuracy_score(y_all, yhat_all)

    out_csv = os.path.join(base_dir, "results_per_person.csv")
    res.to_csv(out_csv, index=False)

    print("")
    print("Saved models to:", os.path.join(base_dir, args.out_models))
    print("Saved per-person results to:", out_csv)
    print("")
    print("Macro avg  FAR:", round(macro_far, 6), "FRR:", round(macro_frr, 6), "ACC:", round(macro_acc, 6))
    print("Micro avg  FAR:", round(micro_far, 6), "FRR:", round(micro_frr, 6), "ACC:", round(micro_acc, 6))
    print("Global CM  TN:", int(tn), "FP:", int(fp), "FN:", int(fn), "TP:", int(tp))


if __name__ == "__main__":
    main()
