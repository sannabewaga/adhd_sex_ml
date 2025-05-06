# ADHD & Sex Prediction Using Logistic Regression

## 📌 Overview

This project explores the application of **logistic regression** on high-dimensional behavioral data to predict two targets:

* **ADHD Diagnosis** (Binary Classification)
* **Sex Classification** (Binary Classification)

Due to the input feature size exceeding 20,000 dimensions, a traditional user-input interface was not feasible. Instead, this Flask web app showcases model performance and insights based on trained data.

---

## 📊 Dataset

* **Samples**: 243 test samples (after split)
* **Features**: 20,000+ behavioral/psychological inputs
* **Targets**:

  * `ADHD` ➔ 0 = No ADHD, 1 = ADHD
  * `Sex` ➔ Binary (0/1)

---

## 🧪 Methodology

* **Algorithm**: Logistic Regression (for both models)
* **Preprocessing**: Vectorization, scaling, and train-test split
* **Evaluation Metrics**:

  * Accuracy
  * Precision
  * Recall
  * F1-Score
  * Confusion Matrix

---

## 📊 Model Performance

### 🪠 ADHD Model

* **Accuracy**: 66%
* **Classification Report**:

```
              precision    recall  f1-score   support

         0.0       0.46      0.40      0.43        77
         1.0       0.74      0.78      0.76       166

    accuracy                           0.66       243
   macro avg       0.60      0.59      0.60       243
weighted avg       0.65      0.66      0.66       243
```

### ♅️ Sex Model

* **Accuracy**: 75%
* **Confusion Matrix**:

```
[[139  29]
 [ 31  44]]
```

* **Classification Report**:

```
              precision    recall  f1-score   support

         0.0       0.82      0.83      0.82       168
         1.0       0.60      0.59      0.59        75

    accuracy                           0.75       243
   macro avg       0.71      0.71      0.71       243
weighted avg       0.75      0.75      0.75       243
```

---

## 🛍️ App Design (Flask)

* App does not accept user input
* Pages:

  * `Home`
  * `Model Performance`
  * `Insights`
  * `Limitations`
* All visualizations and metrics shown statically

---

## 🔍 Insights

* **ADHD Model**:

  * High recall for ADHD cases
  * Many false positives for non-ADHD
  * Useful as a screening tool, not for diagnosis

* **Sex Model**:

  * Higher accuracy for majority class
  * Lower performance for minority class
  * May reflect behavioral patterns

---

## ⚠️ Limitations

* Very high feature dimensionality
* Models trained and validated only on internal dataset
* ADHD model underperforms on negative class
* No generalizability to clinical usage
* Potential ethical implications in usage

---

---

## 🚀 How to Run

```bash
pip install -r requirements.txt
python app.py
```

Then open your browser to `http://127.0.0.1:5000`

---

## 🚧 Disclaimer

This project is for academic demonstration purposes only. It is **not** intended for clinical or diagnostic use.

---


