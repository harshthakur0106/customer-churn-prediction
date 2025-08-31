# Customer Churn Prediction using Machine Learning

## 📌 Project Overview
This project focuses on predicting **customer churn** for a telecom company using machine learning models. Customer churn refers to when a customer stops using a company's services, and predicting this behavior helps businesses take preventive actions.

---

## 📂 Dataset
The dataset used in this project is:
- **Telco Customer Churn Dataset**
- Key features include customer demographics, account information, and service usage patterns.
- Target variable: `Churn` (Yes/No)

---

## 🔍 Key Steps in the Notebook
1. **Importing Dependencies**  
   Libraries: `NumPy`, `Pandas`, `Matplotlib`, `Seaborn`, `Scikit-learn`, `XGBoost`, `imblearn`

2. **Data Loading and Preprocessing**
   - Removed unnecessary columns (e.g., Customer ID)
   - Handled missing values in `TotalCharges`
   - Addressed **class imbalance** using **SMOTE** (Synthetic Minority Over-sampling Technique)
   - Label encoding for categorical variables

3. **Exploratory Data Analysis (EDA)**
   - Distribution of churn vs non-churn customers
   - Analysis of numerical and categorical features
   - Correlation analysis

4. **Model Building**
   - **Decision Tree Classifier**
   - **Random Forest Classifier**
   - **XGBoost Classifier**

5. **Model Evaluation**
   - Accuracy Score
   - Confusion Matrix
   - Classification Report

6. **Model Saving**
   - Best performing model saved using **Pickle** for future predictions

---

## 🛠️ Requirements
Install the following dependencies:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn imbalanced-learn xgboost

▶️ How to Run the Project
1. Clone the Repository
git clone <your-repository-url>
cd <your-repository-folder>

2. Prepare the Dataset

Download the dataset: WA_Fn-UseC_-Telco-Customer-Churn.csv

Place it in the root directory or update the file path in the notebook.

3. Open the Jupyter Notebook
jupyter notebook Copy_of_Customer_Churn_Prediction_using_ML.ipynb

4. Run All Cells

The notebook will:

Load and preprocess the data

Handle missing values and class imbalance

Perform EDA (Exploratory Data Analysis)

Train ML models (Decision Tree, Random Forest, XGBoost)

Evaluate the models using Accuracy, Confusion Matrix, and Classification Report

Save the best model using Pickle

