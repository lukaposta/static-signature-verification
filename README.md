# Static Signature Verification (Writer-Dependent)

This repository contains a complete implementation of a **writer-dependent static signature verification system**, including preprocessing, feature extraction, per-user machine learning models, evaluation, and a simple local web application for interactive verification.

The project was developed as part of an academic course in biometric systems and focuses on classical machine learning techniques rather than deep learning.

---

## Project Overview

The goal of this project is to verify whether a given **offline (static) handwritten signature** belongs to a claimed individual.

The system is **writer-dependent**, meaning that a separate binary classifier is trained for each enrolled person.

### Verification pipeline

1. Image preprocessing  
2. Feature extraction using Histogram of Oriented Gradients (HOG)  
3. Writer-dependent classification using Support Vector Machines (SVM)  
4. Decision based on SVM decision score and a fixed threshold  
5. Presentation of the result via a local web interface  

---

## Key Characteristics

- Offline (static) signature verification  
- Writer-dependent approach (one model per person)  
- Classical machine learning (no neural networks)  
- Interpretable pipeline  
- Local web application for demonstration and testing  

---

## Dataset

Experiments were conducted using the **CEDAR signature dataset**.

Dataset files are **not included** in this repository due to size and licensing constraints.

The expected dataset structure is:

```

dataset55/
├── 001/
│   ├── genuine signatures
├── 001_forg/
│   ├── forged signatures
├── 002/
├── 002_forg/
...

```

A CSV index (`data.csv`) is generated to manage train/test splits.

---

## Preprocessing

Each signature image is processed using the following steps:

1. Grayscale conversion  
2. Binarization (Otsu or adaptive thresholding)  
3. Optional inversion to ensure consistent foreground representation  
4. Morphological opening to remove noise  
5. Bounding box extraction around the signature  
6. Resizing to a fixed resolution  

This ensures consistent input dimensions and reduces background variability.

---

## Feature Extraction

Histogram of Oriented Gradients (HOG) features are extracted from the preprocessed images.

HOG was chosen because it:
- captures local stroke orientation patterns  
- is robust to small geometric variations  
- works well with limited training samples  

---

## Classification Model

For each enrolled person, a **binary Support Vector Machine (SVM)** is trained:

- Positive class: genuine signatures of the target person  
- Negative class: genuine signatures of other persons (writer-dependent setup)  
- Kernel: RBF  
- Class balancing enabled  

Each model outputs a **decision score**, not a probability.

---

## Decision and Confidence

### Decision rule

A signature is accepted if:

```

decision_score ≥ threshold

```

By default, the threshold is set to `0.0`, which corresponds to the natural SVM decision boundary.

### Confidence value

The displayed confidence is **not a probability**.

It is computed by applying a **sigmoid transformation** to the SVM decision score:

```

p = 1 / (1 + exp(-score))

```

- If the decision is ACCEPT: `confidence = p × 100%`
- If the decision is REJECT: `confidence = (1 − p) × 100%`

This value represents a **scaled distance from the decision boundary**, not a calibrated likelihood.

---

## Web Application

A lightweight **Flask-based web application** is provided for interactive testing.

### Features

- Selection of a person-specific model  
- Upload of a signature image (genuine or forged)  
- Display of:
  - ACCEPT / REJECT decision  
  - Confidence score  
  - Original uploaded image  
  - Preprocessed (binarized) image  

### Example results


**Accepted genuine signature**

![Accept example](https://github.com/user-attachments/assets/54ea7afc-e464-4840-b624-791fbf1ba85e)


**Rejected forged signature**

![Reject example](https://github.com/user-attachments/assets/28dfc0f7-c9d5-4872-bb30-83f3eaa04ab0)


---

## Evaluation

Evaluation is performed on a held-out test set and aggregated across all persons.

### Confusion matrix (aggregated test set)

<p>
  <img src="https://github.com/user-attachments/assets/9a774d11-9cd6-4d15-9b98-8435509bfa11" width="600">
</p>


### Precision–Recall curve

Average Precision (AP) ≈ 0.92

<p>
  <img src="https://github.com/user-attachments/assets/148bd542-d4e6-49e5-ac2b-e610bb4d1efa" width="600">
</p>


### ROC curve

ROC AUC ≈ 0.97

<p>
  <img src="https://github.com/user-attachments/assets/bdb8d02c-bb8b-4bb4-8fd9-6d397a46e54f" width="600">
</p>


These results indicate strong separability between genuine and forged signatures in the chosen feature space.

---

## Repository Structure

```

code/
├── app.py                  # Flask web application
├── preprocess.py           # Image preprocessing pipeline
├── verify_one.py            # CLI verification script
├── hog_svm_train_eval.py    # Training and evaluation
├── plot_roc_pr.py           # ROC / PR plotting
├── make_csv.py              # Dataset CSV generation
├── templates/
│   └── index.html           # Web UI
├── results_per_person.csv   # Evaluation summary

```

---

## Installation

Python 3.10 or newer is recommended.

Install dependencies:

```

pip install -r requirements.txt

```

---

## Running the Web Application

From the `code` directory:

```

python app.py

```

Then open: <a href="http://127.0.0.1:5000" target="_blank">http://127.0.0.1:5000</a>



Trained models must be placed in:

```

code/models/svm_person_{id}.joblib

```

---

## Notes on Reproducibility

- Dataset and trained models are not included  
- Random seeds are fixed where applicable  
- The pipeline is deterministic given identical inputs  

---

## License

This project is released under the **MIT License**.

---

## Academic Context

This project was developed for academic purposes and demonstrates a complete biometric verification pipeline using classical machine learning methods. It is intended for educational and research use rather than deployment in real-world security systems.
