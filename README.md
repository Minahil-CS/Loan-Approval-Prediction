# 🚀 Loan Approval Prediction System

An end-to-end Machine Learning web-ready application designed to automate and streamline the credit risk assessment process using supervised binary classification (**Logistic Regression**).

---

## 📋 Table of Contents
1. [About The Project](#-about-the-project)
2. [Dataset Overview](#-dataset-overview)
3. [Project Workflow & Architecture](#-project-workflow--architecture)
4. [Exploratory Data Analysis (EDA) Insights](#-exploratory-data-analysis-eda-insights)
5. [Model Performance & Evaluation](#-model-performance--evaluation)
6. [Tech Stack](#-tech-stack)
7. [Getting Started & Installation](#-getting-started--installation)
8. [Usage & Predictions](#-usage--predictions)

---

## 🎯 About The Project
Traditional loan approval workflows in financial institutions are often manual, time-consuming, and susceptible to subjective bias. This project implements a machine learning pipeline that predicts whether an applicant’s loan request should be **Approved (Y)** or **Rejected (N)** based on their financial history, demographics, and loan parameters. 

---

## 📊 Dataset Overview
The model is trained on a standardized financial dataset containing historical loan application records. Key features include:
* **Financial Attributes:** `ApplicantIncome`, `CoapplicantIncome`, `LoanAmount`, `Loan_Amount_Term`
* **Credit History:** `Credit_History` (Binary indicator of whether guidelines were met)
* **Demographics & Property:** `Gender`, `Married`, `Dependents`, `Education`, `Self_Employed`, `Property_Area`
* **Target Variable:** `Loan_Status` (Y = Approved, N = Rejected)

---

## ⚙️ Project Workflow & Architecture
1. **Data Preprocessing & Cleaning:** 
   * Handled missing values using statistical imputation (Mode for categorical attributes, Median for continuous variables like `LoanAmount`).
   * Dropped redundant unique identifiers (`Loan_ID`).
   * Encoded categorical attributes into numerical values for ML compatibility and applied `pd.get_dummies` for multi-category features.
2. **Exploratory Data Analysis (EDA):** Visualized distribution patterns, income skewness, and feature-to-target correlations using `Matplotlib` and `Seaborn`.
3. **Data Splitting & Scaling:** Divided the dataset into an 80-20 train-test split and normalized feature scales using `StandardScaler` to optimize model convergence.
4. **Model Training:** Implemented a robust **Logistic Regression** classifier.
5. **Evaluation & Prediction Pipeline:** Tested against multiple unseen applicant profiles to output risk classes and confidence probabilities.

---

## 📈 Exploratory Data Analysis (EDA) Insights
* **Credit History Impact:** Applicants with a clean credit history (`Credit_History = 1.0`) demonstrate a dramatically higher approval rate, making it the single most influential predictive feature.
* **Income Distribution:** Both applicant and co-applicant incomes exhibit right-skewness, handled effectively during scaling.
* **Property Area Trends:** Semi-urban locations show balanced distribution with favorable approval ratios compared to strictly rural or urban zones.

---

## 📊 Model Performance & Evaluation
The trained Logistic Regression model achieved strong metrics on the test dataset:
* **Accuracy:** ~80%+
* **Evaluation Metrics Tracked:** Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC score confirming high classification reliability.

---

## 🛠️ Tech Stack
* **Language:** Python 3.x
* **Libraries:** Pandas, NumPy, Scikit-Learn, Matplotlib, Seaborn
* **Environment:** Jupyter Notebook / JupyterLab
* **Version Control:** Git & GitHub

---

## 🚀 Getting Started & Installation

To run this project locally on your machine, follow these steps:

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/Minahil-CS/Loan-Approval-Prediction.git](https://github.com/Minahil-CS/Loan-Approval-Prediction.git)
~~

2.   Navigate to the project directory:
```
Bash
cd Loan-Approval-Prediction
Install required dependencies:
```

3. Install required dependencies:
```
Bash
pip install pandas numpy scikit-learn matplotlib seaborn
````
4. Run the Jupyter Notebook:
Open Loan Prediction.ipynb inside JupyterLab or VS Code and execute the cells sequentially.

💡 Usage & Predictions
The system includes a custom multi-instance inference wrapper. You can pass unformatted applicant arrays to output instant approval outcomes and confidence percentages:

Python
# Example sample input structure for real-time predictions
test_applicants = [...] 
# Outputs prediction class and confidence probability score
