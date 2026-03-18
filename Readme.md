# Heart Disease Prediction using Machine Learning and Deep Learning

## Abstract

Cardiovascular diseases remain one of the leading causes of mortality worldwide. Early detection using clinical data can significantly improve patient outcomes. This project presents a comparative study between a classical machine learning model (Logistic Regression) and a deep learning model (Feedforward Neural Network) for predicting the presence of heart disease using the UCI Heart Disease dataset. The study demonstrates that a neural network can capture nonlinear relationships in clinical data and achieve improved predictive performance.

---

## 1. Introduction

Heart disease prediction is a critical problem in biomedical data science. Clinical datasets often contain multiple physiological and diagnostic variables that contribute to disease risk. The goal of this project is to develop a predictive model that can classify whether a patient has heart disease based on such features.

This project follows a structured machine learning pipeline:

* Data preprocessing and cleaning
* Baseline model implementation
* Neural network modeling
* Performance evaluation and comparison

---

## 2. Problem Definition

### Task

Binary classification

### Objective

Predict whether a patient has heart disease.

### Target Variable

* `0` → No heart disease
* `1` → Heart disease present

### Input Features

Clinical and physiological attributes including:

* Age
* Sex
* Chest pain type
* Resting blood pressure
* Cholesterol level
* Fasting blood sugar
* ECG results
* Maximum heart rate
* Exercise-induced angina
* ST depression and slope
* Number of vessels
* Thalassemia status

---

## 3. Dataset Description

* Source: UCI Heart Disease Dataset (Cleveland subset)
* Samples: ~300
* Features: 13
* Original target: multi-class (0–4 severity levels)
* Converted to binary classification:

  * `0` → no disease
  * `1–4` → disease present

### Data Cleaning

* Missing values represented as `"?"` were converted to `NaN`
* Rows with missing values were removed
* All features converted to numeric format

---

## 4. Methodology

### 4.1 Data Preprocessing

* Stratified splitting:

  * 70% training
  * 15% validation
  * 15% test
* Feature scaling using StandardScaler
* Ensured no data leakage (scaler fit only on training data)

---

### 4.2 Baseline Model: Logistic Regression

A logistic regression model was used as a baseline due to its effectiveness on tabular medical data.

#### Results (Validation Set)

* Accuracy: 0.83
* F1 Score: 0.81
* ROC-AUC: 0.90

---

### 4.3 Neural Network Model

A feedforward neural network was implemented using PyTorch.

#### Architecture

* Input layer: 13 features
* Hidden Layer 1: 32 neurons + ReLU + Dropout (0.2)
* Hidden Layer 2: 16 neurons + ReLU + Dropout (0.2)
* Output Layer: 1 neuron (logits)

#### Training Details

* Loss Function: Binary Cross Entropy with Logits (BCEWithLogitsLoss)
* Optimizer: Adam
* Learning Rate: 0.001
* Epochs: 50
* Batch Size: 32

---

## 5. Training Analysis

During training, the model showed:

* Decreasing training and validation loss initially
* Validation loss plateauing after ~25–30 epochs
* Increasing validation loss afterward

This indicates **overfitting**.

To address this:

* The best model was saved based on **minimum validation loss** (early stopping approach)

---

## 6. Results

### Final Model Performance (Test Set)

| Metric   | Value    |
| -------- | -------- |
| Accuracy | **0.90** |
| F1 Score | **0.89** |
| ROC-AUC  | **0.93** |

---

## 7. Comparison with Baseline

| Model               | Accuracy | F1 Score | ROC-AUC  |
| ------------------- | -------- | -------- | -------- |
| Logistic Regression | 0.83     | 0.81     | 0.90     |
| Neural Network      | **0.90** | **0.89** | **0.93** |

### Key Insight

The neural network outperformed logistic regression, suggesting the presence of **nonlinear relationships** in the dataset that are not captured by linear models.

---

## 8. Debugging and Challenges

During development, a key issue was encountered:

* **Tensor shape mismatch**

  * Model output shape: `[batch_size, 1]`
  * Target shape: `[batch_size]`

### Solution

Reshaped target tensor using:

```
.view(-1, 1)
```

This ensured compatibility with the loss function.

---

## 9. Discussion

* Logistic regression performed strongly, indicating partially linear separability.
* Neural network provided consistent improvements across all metrics.
* ROC-AUC improvement indicates better ranking of patient risk.
* Overfitting behavior highlights the importance of validation monitoring.

---

## 10. Future Work

With more time, the following improvements can be explored:

* Hyperparameter tuning (learning rate, architecture)
* Cross-validation for robust evaluation
* Feature importance analysis
* Model interpretability (SHAP, LIME)
* Testing on larger or multi-center datasets

---

## 11. How to Run

### Install dependencies

```
pip install -r requirements.txt
```

### Train model

```
python -m src.train
```

### Evaluate model

```
python -m src.evaluate
```

### Run baseline

```
python -m src.baseline
```

---

## 12. Project Structure

```
heart-disease-pytorch/
│
├── data/
│   ├── raw/
│   └── processed/
├── src/
│   ├── data_preprocessing.py
│   ├── dataset.py
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── baseline.py
├── models/
├── results/
├── notebooks/
└── README.md
```

---

## 13. Conclusion

This project demonstrates a complete machine learning pipeline for biomedical classification. The neural network model successfully improved upon a strong baseline, highlighting the importance of model selection and validation. The work emphasizes not only performance but also proper engineering practices, debugging, and interpretability considerations in real-world ML problems.

---
