# Employee Attrition Prediction

A production-ready Machine Learning web application that predicts whether an employee is likely to leave an organization using multiple classification algorithms and an end-to-end Scikit-Learn pipeline.

**Developed by:** Mahi Agrawal

![Python](https://img.shields.io/badge/Python-3.10+-blue?logo=python)
![Flask](https://img.shields.io/badge/Flask-2.3+-green?logo=flask)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.3+-orange?logo=scikit-learn)
![Bootstrap](https://img.shields.io/badge/Bootstrap-5-purple?logo=bootstrap)

---

# Project Overview

Employee attrition is a major challenge for organizations because replacing experienced employees is both time-consuming and expensive. This project leverages the **IBM HR Analytics Employee Attrition & Performance** dataset to build an intelligent prediction system that identifies employees who are likely to leave the company.

The application compares multiple machine learning models, automatically selects the best-performing model, and serves predictions through an interactive Flask web interface.

---

# Highlights

- End-to-end Machine Learning pipeline
- Automated model comparison
- Pipeline-based preprocessing
- Interactive Flask web application
- Feature importance visualization
- Responsive Bootstrap interface
- Confidence score prediction
- Clean production-style project structure

---

# Features

- **End-to-End ML Pipeline** using Scikit-Learn Pipeline and ColumnTransformer
- **Automatic Model Comparison**
  - Logistic Regression
  - Random Forest
  - XGBoost
  - CatBoost
- **Comprehensive Evaluation**
  - Accuracy
  - Precision
  - Recall
  - F1 Score
  - ROC-AUC
  - Confusion Matrix
  - Classification Report
- **Feature Importance Visualization**
- **Interactive Prediction Interface**
- **Reusable Production Pipeline**
- **Clean Flask Architecture**

---

# Tech Stack

| Component | Technology |
|-----------|------------|
| Language | Python |
| Machine Learning | Scikit-Learn, XGBoost, CatBoost |
| Backend | Flask |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| Serialization | Joblib |
| Frontend | HTML, CSS, Bootstrap |


---

# Folder Structure

```text
Employee-Attrition-Prediction/
│
├── app.py
├── train.py
├── utils.py
├── pipeline.pkl
├── requirements.txt
├── README.md
├── LICENSE
├── SECURITY.md
├── Attrition.ipynb
│
├── data/
│   └── WA_Fn-UseC_-HR-Employee-Attrition.csv
│
├── static/
│   └── feature_importance.png
│
└── templates/
    └── index.html
```

---



# Usage

1. Open the web application.
2. Enter employee information.
3. Click **Predict**.
4. View the predicted attrition result along with confidence score.

---

# Model Performance

The project compares multiple machine learning models using **5-fold Stratified Cross-Validation** on the training data and selects the best model based on the mean **ROC-AUC** score.

| Model | CV ROC-AUC |
|--------|------------|
| **Logistic Regression** | **82.65%** |
| Random Forest | 82.38% |
| CatBoost | 82.14% |
| XGBoost | 80.75% |

### Best Model

**Logistic Regression** achieved the highest mean **CV ROC-AUC of 82.65%** and was selected as the final model.

### Final Test-Set Performance

| Metric | Score |
|--------|-------|
| Accuracy | **76.87%** |
| Precision | **37.65%** |
| Recall | **68.09%** |
| F1-Score | **48.48%** |
| ROC-AUC | **80.39%** |
---

# Screenshots

## Home Page

| Homepage 1 | Homepage 2 |
|------------|------------|
| ![Homepage 1](screenshots/homepage1.png) | ![Homepage 2](screenshots/homepage2.png) |

---

## Prediction Results

| Employee Likely to Stay | Employee Likely to Leave |
|--------------------------|--------------------------|
| ![Stay Prediction](screenshots/stay_prediction.png) | ![Leave Prediction](screenshots/leave_prediction.png) |

---

# Project Workflow

```text
Employee Information
        │
        ▼
Flask Web Application
        │
        ▼
Scikit-Learn Pipeline
        │
        ├── Feature Engineering
        ├── Column Transformer
        ├── One-Hot Encoding
        └── Best ML Model
                │
                ▼
Employee Attrition Prediction
```

---

# Skills Demonstrated

- Machine Learning
- Feature Engineering
- Data Preprocessing
- Classification Algorithms
- Model Evaluation
- Flask Development
- Scikit-Learn Pipelines
- Joblib Serialization
- Data Visualization
- Bootstrap UI Development

---

# Future Improvements

- Deploy using Render or AWS
- Explain predictions using SHAP
- Store prediction history in a database
- Export prediction reports


---

# Dataset

IBM HR Analytics Employee Attrition & Performance Dataset

- 1470 employee records
- 35 employee-related features
- Binary classification problem


