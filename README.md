# 🧬 Breast Cancer Prediction Using Logistic Regression

This project is part of the AI & ML Internship Task 4, focused on building a **binary classification model** using **Logistic Regression**. The goal is to predict whether a tumor is malignant or benign based on features extracted from cell nuclei in breast mass biopsies.

---

## 🎯 Objective

- Implement logistic regression for binary classification
- Standardize and split dataset into training and testing sets
- Evaluate performance using metrics like confusion matrix, precision, recall, F1-score, ROC-AUC
- Visualize results and understand the sigmoid function's role

---

## 📁 Dataset

- **Source:** [Breast Cancer Wisconsin Dataset – Kaggle](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)
- **Size:** 569 samples, 32 columns (ID, diagnosis label, and 30 numerical features)
- **Target:** `diagnosis`  
  - M (Malignant) → 1  
  - B (Benign) → 0

---

## 🛠 Tools & Libraries

- **Python**
- **Jupyter Notebook**
- `pandas`, `numpy` — Data manipulation
- `matplotlib`, `seaborn` — Visualization
- `scikit-learn` — Modeling and evaluation

---

## 🚀 Workflow Summary

### 1. **Data Loading & Cleaning**
- Removed ID and unnamed columns
- Encoded target labels (M → 1, B → 0)
- Checked for null values and outliers

### 2. **Feature Scaling**
- Standardized numerical features using `StandardScaler` for better model performance

### 3. **Train-Test Split**
- Used 80-20 split via `train_test_split`

### 4. **Model Training**
- Fitted a `LogisticRegression` model using scikit-learn

### 5. **Prediction & Evaluation**
- Evaluated using:
  - **Confusion Matrix**
  - **Classification Report**
  - **ROC-AUC Score**
  - **ROC Curve Visualization**

### 6. **Threshold Tuning (Optional)**
- Demonstrated how changing the decision threshold affects predictions

---

## 📊 Results

| Metric           | Score     |
|------------------|-----------|
| Accuracy         | 0.96      |
| Precision        | 0.95      |
| Recall           | 0.98      |
| F1-score         | 0.96      |
| ROC-AUC Score    | 0.99      |

> The model shows high predictive performance and generalizes well on test data.

---

## 🧠 Key Concepts

- **Logistic Regression** is used for predicting binary outcomes
- **Sigmoid Function** squashes the output to a 0–1 probability
- **Confusion Matrix** helps understand TP, TN, FP, FN
- **ROC-AUC Curve** shows tradeoff between TPR and FPR

---

## 📂 Project screenshot

![Screenshot 2025-06-17 230326](https://github.com/user-attachments/assets/1a80d043-db0f-4c83-93f7-0de2bb29bbea)
