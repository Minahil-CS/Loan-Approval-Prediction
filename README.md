<div align="center">

# 🏦 Loan Approval Prediction System

**An end-to-end Machine Learning pipeline that predicts whether a loan application will be approved, using applicant details and Logistic Regression.**

![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-ML-F7931E?logo=scikitlearn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-Data%20Analysis-150458?logo=pandas&logoColor=white)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-F37626?logo=jupyter&logoColor=white)
![Accuracy](https://img.shields.io/badge/Accuracy-78.86%25-success)
![Status](https://img.shields.io/badge/Status-Complete-brightgreen)

</div>

---

## 📋 Table of Contents

1. [About the Project](#-about-the-project)
2. [Key Results](#-key-results)
3. [Dataset Overview](#-dataset-overview)
4. [Project Workflow](#-project-workflow)
5. [EDA Insights](#-exploratory-data-analysis-eda-insights)
6. [Model Performance](#-model-performance--evaluation)
7. [Predicting New Applicants](#-predicting-new-applicants)
8. [Tech Stack](#-tech-stack)
9. [Project Structure](#-project-structure)
10. [Getting Started](#-getting-started)
11. [Limitations and Future Work](#-limitations-and-future-work)
12. [Author](#-author)

---

## 🎯 About the Project

Banks and lenders receive thousands of loan applications, and reviewing each one manually is slow and can be inconsistent. This project builds a supervised **binary classification** model that predicts whether a loan application should be **Approved (Y)** or **Rejected (N)** from the applicant's profile, supporting faster and more data-driven credit decisions.

The project covers the complete machine learning workflow: data loading, cleaning, preprocessing, exploratory analysis, model training, evaluation, and inference on new applicants.

## 🏆 Key Results

| Metric | Score |
|---|---|
| **Accuracy** | 78.86% |
| **Precision** | 75.96% |
| **Recall** | 98.75% |
| **F1-Score** | 85.87% |
| **ROC-AUC** | 74.04% |
| Training Accuracy | 82.08% |

> 💡 **Key insight:** `Credit_History` is the strongest factor influencing the loan decision (correlation of **0.54** with `Loan_Status`).

## 📊 Dataset Overview

- **File:** `train_u6lujuX_CVtuZ9i.csv`
- **Size:** 614 applications, 13 columns
- **Target:** `Loan_Status` (`Y` = Approved, `N` = Rejected)

| Feature | Description |
|---|---|
| `Loan_ID` | Unique ID (dropped before training) |
| `Gender` | Male / Female |
| `Married` | Marital status |
| `Dependents` | Number of dependents (0, 1, 2, 3+) |
| `Education` | Graduate / Not Graduate |
| `Self_Employed` | Self-employment status |
| `ApplicantIncome` | Applicant's income |
| `CoapplicantIncome` | Co-applicant's income |
| `LoanAmount` | Requested loan amount |
| `Loan_Amount_Term` | Loan term in months |
| `Credit_History` | Credit history meets guidelines (1) or not (0) |
| `Property_Area` | Urban / Semiurban / Rural |
| `Loan_Status` | Target variable |

## 🔄 Project Workflow

```
Raw Data → Cleaning → Encoding → EDA → Train/Test Split → Scaling → Logistic Regression → Evaluation → Predictions
```

1. **Data cleaning**
   - Dropped the non-informative `Loan_ID` column.
   - Filled missing categorical and discrete values (`Gender`, `Married`, `Dependents`, `Self_Employed`, `Loan_Amount_Term`, `Credit_History`) with the **mode**.
   - Filled missing `LoanAmount` values with the **median**.
2. **Encoding**
   - Binary mapping for `Gender`, `Married`, `Education`, `Self_Employed` and `Loan_Status`.
   - Converted `Dependents` (`3+` to `3`) to a numeric type.
   - One-hot encoded `Property_Area` (`drop_first=True`), giving **12 model features**.
3. **Exploratory Data Analysis** with Matplotlib and Seaborn.
4. **Train/test split:** 80% / 20% (`random_state=42`).
5. **Feature scaling** with `StandardScaler` (fit on training data only to avoid data leakage).
6. **Model training:** Logistic Regression.
7. **Evaluation** with a confusion matrix, classification report and ROC-AUC.
8. **Inference function** that returns the prediction and a confidence score for new applicants.

## 🔍 Exploratory Data Analysis (EDA) Insights

- **Credit History** is the dominant signal: applicants with a credit history of `1` are approved far more often than those with `0`.
- **Applicant Income** is heavily right-skewed, with most applicants in the lower income range and a few high-income outliers.
- **Correlation matrix:** `Credit_History` has the strongest correlation with `Loan_Status` (**0.54**), and `LoanAmount` correlates with `ApplicantIncome` (**0.57**).

## 📈 Model Performance & Evaluation

**Confusion Matrix** (test set, 123 samples)

|  | Predicted Rejected | Predicted Approved |
|---|:---:|:---:|
| **Actual Rejected** | 18 | 25 |
| **Actual Approved** | 1 | 79 |

**Classification Report**

| Class | Precision | Recall | F1-Score | Support |
|---|:---:|:---:|:---:|:---:|
| 0 (Rejected) | 0.95 | 0.42 | 0.58 | 43 |
| 1 (Approved) | 0.76 | 0.99 | 0.86 | 80 |
| **Accuracy** | | | **0.79** | 123 |

**Interpretation:** the model is excellent at identifying approvable applicants (99% recall) but more lenient on rejections (42% recall for the rejected class), which is a result of the class imbalance in the data.

## 🔮 Predicting New Applicants

The notebook includes a reusable `predict_loan_status()` function that scales new inputs with the fitted scaler and returns the decision with a confidence probability.

| Applicant | Prediction | Confidence |
|---|---|---|
| 1 | Approved (Y) | 90.16% |
| 2 | Rejected (N) | 95.10% |
| 3 | Approved (Y) | 81.08% |

## 🛠️ Tech Stack

- **Language:** Python 3
- **Data processing:** Pandas, NumPy
- **Visualization:** Matplotlib, Seaborn
- **Machine learning:** Scikit-Learn (Logistic Regression, StandardScaler, metrics)
- **Environment:** Jupyter Notebook

## 📁 Project Structure

```
Loan-Approval-Prediction/
├── Loan Prediction.ipynb        # Full analysis, model training and evaluation
├── train_u6lujuX_CVtuZ9i.csv    # Dataset
├── requirements.txt             # Python dependencies
└── README.md                    # Project documentation
```

## 🚀 Getting Started

**1. Clone the repository**

```bash
git clone https://github.com/Minahil-CS/Loan-Approval-Prediction.git
cd Loan-Approval-Prediction
```

**2. (Optional) Create a virtual environment**

```bash
python -m venv venv
source venv/bin/activate        # On Windows: venv\Scripts\activate
```

**3. Install dependencies**

```bash
pip install -r requirements.txt
```

**4. Launch the notebook**

```bash
jupyter notebook "Loan Prediction.ipynb"
```

Keep `train_u6lujuX_CVtuZ9i.csv` in the same folder as the notebook, then run all cells from top to bottom.

## ⚠️ Limitations and Future Work

- Class imbalance lowers recall on rejected applications; techniques like **SMOTE**, class weights or threshold tuning can help.
- Compare with other models (**Random Forest, XGBoost, SVM**) and use cross-validation and hyperparameter tuning.
- Add feature engineering (e.g. total income, EMI-to-income ratio).
- Build and deploy an interactive **Streamlit** web app for live predictions.

## 👩‍💻 Author

**Minahil Fatima**
BS Computer Science, University of Agriculture Faisalabad (UAF)

[![GitHub](https://img.shields.io/badge/GitHub-Minahil--CS-181717?logo=github)](https://github.com/Minahil-CS)

Built as part of the AI/ML Internship at **Big Brains**.

---

<div align="center">

⭐ If you found this project useful, consider giving it a star!

</div>
