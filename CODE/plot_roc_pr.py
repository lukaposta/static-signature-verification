import os
import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from sklearn.metrics import (
    roc_curve, roc_auc_score,
    precision_recall_curve, average_precision_score,
    confusion_matrix
)

def main():
    base_dir = os.path.dirname(os.path.abspath(__file__))

    # Mora postojati cache od HOG-a i modeli iz treninga
    cache_path = os.path.join(base_dir, "cache", "features_hog.npz")
    models_dir = os.path.join(base_dir, "models")
    out_dir = os.path.join(base_dir, "plots")
    os.makedirs(out_dir, exist_ok=True)

    if not os.path.exists(cache_path):
        raise FileNotFoundError(f"Nema cachea: {cache_path}. Pokreni hog_svm_train_eval.py prvo.")
    if not os.path.isdir(models_dir):
        raise FileNotFoundError(f"Nema models foldera: {models_dir}. Pokreni hog_svm_train_eval.py prvo.")

    data = np.load(cache_path, allow_pickle=True)
    X = data["X"]
    df = pd.DataFrame({
        "image_path": data["image_path"],
        "person_id": data["person_id"],
        "label": data["label"],
        "split": data["split"],
    })

    # Test uzorci
    test_g = df[(df["split"] == "test") & (df["label"] == "genuine")]
    test_f = df[(df["split"] == "test") & (df["label"] == "forged")]

    persons = np.sort(df[(df["split"] == "train") & (df["label"] == "genuine")]["person_id"].unique())

    y_true_all = []
    scores_all = []
    y_pred_all = []

    missing_models = 0
    used_persons = 0

    for pid in persons:
        model_path = os.path.join(models_dir, f"svm_person_{pid}.joblib")
        if not os.path.exists(model_path):
            missing_models += 1
            continue

        pack = joblib.load(model_path)
        clf = pack["model"]
        thr = float(pack.get("threshold", 0.0))

        pos_idx = test_g.index[test_g["person_id"].to_numpy() == pid].to_numpy()
        neg_idx = test_f.index[test_f["person_id"].to_numpy() == pid].to_numpy()

        if len(pos_idx) == 0 or len(neg_idx) == 0:
            continue

        X_pos = X[pos_idx]
        X_neg = X[neg_idx]
        X_test = np.vstack([X_pos, X_neg])

        y_true = np.hstack([
            np.ones((len(X_pos),), dtype=int),
            np.zeros((len(X_neg),), dtype=int)
        ])

        # decision_function score (veći score = više "genuine")
        scores = clf.decision_function(X_test)

        # binarna odluka na pragu 0 (isti princip kao u trening skripti)
        y_pred = (scores >= thr).astype(int)

        y_true_all.append(y_true)
        scores_all.append(scores)
        y_pred_all.append(y_pred)
        used_persons += 1

    if used_persons == 0:
        raise RuntimeError("Nisam uspio skupiti test scoreove ni za jednu osobu. Provjeri modele i cache.")

    y_true_all = np.concatenate(y_true_all)
    scores_all = np.concatenate(scores_all)
    y_pred_all = np.concatenate(y_pred_all)

    # ROC + AUC
    fpr, tpr, _ = roc_curve(y_true_all, scores_all)
    auc = roc_auc_score(y_true_all, scores_all)

    plt.figure()
    plt.plot(fpr, tpr)
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("FPR (FAR)")
    plt.ylabel("TPR")
    plt.title(f"ROC krivulja (AUC={auc:.3f})")
    roc_path = os.path.join(out_dir, "roc_curve.png")
    plt.savefig(roc_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Precision-Recall + AP
    precision, recall, _ = precision_recall_curve(y_true_all, scores_all)
    ap = average_precision_score(y_true_all, scores_all)

    plt.figure()
    plt.plot(recall, precision)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title(f"Precision-Recall krivulja (AP={ap:.3f})")
    pr_path = os.path.join(out_dir, "pr_curve.png")
    plt.savefig(pr_path, dpi=200, bbox_inches="tight")
    plt.close()

    # Confusion matrix heatmap (matplotlib)
    tn, fp, fn, tp = confusion_matrix(y_true_all, y_pred_all, labels=[0, 1]).ravel()
    cm = np.array([[tn, fp],
                   [fn, tp]])

    plt.figure()
    plt.imshow(cm)
    for (i, j), v in np.ndenumerate(cm):
        plt.text(j, i, str(v), ha="center", va="center")
    plt.xticks([0, 1], ["Forged", "Genuine"])
    plt.yticks([0, 1], ["Forged", "Genuine"])
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Matrica zabune (agregirano, test)")
    cm_path = os.path.join(out_dir, "confusion_matrix.png")
    plt.savefig(cm_path, dpi=200, bbox_inches="tight")
    plt.close()

    print("Gotovo.")
    print("Osoba korišteno:", used_persons, "| Nedostaje modela:", missing_models)
    print("ROC AUC:", round(float(auc), 6))
    print("PR AP:", round(float(ap), 6))
    print("Spremio u:", out_dir)
    print("TN:", int(tn), "FP:", int(fp), "FN:", int(fn), "TP:", int(tp))

if __name__ == "__main__":
    main()
