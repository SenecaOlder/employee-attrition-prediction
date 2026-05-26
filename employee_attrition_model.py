##### Employee Attrition Prediction — Logistic Regression Model #####
### CSCI-5310 | Joseph Bray Demonbreun
### Dataset: "EDA-Analyzing the Attrition Rate of a Company" by Anuj Biswas (Kaggle)
### https://www.kaggle.com/datasets/anujachintyabiswas/attrition-rate-of-acompany/data

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn import metrics

# ============================================================
# Step 1: Load Dataset
# Download dataset from Kaggle link above and update path.
# ============================================================

text = pd.read_csv('Attrition_data_original.csv')
print(text.shape)       # 4410 rows x 29 columns
print(text.head())

# ============================================================
# Step 2: Data Cleaning & Target Encoding
# - Drop rows with missing values
# - Encode target variable: Yes -> 1, No -> 0
# ============================================================

text.dropna(inplace=True)
text['Attrition'] = text['Attrition'].map({'Yes': 1, 'No': 0})

print("\nClass distribution (imbalanced):")
print(text['Attrition'].value_counts())
# Result: 3605 stayed (0), 695 left (1) — 84:16 imbalance

# ============================================================
# Step 3: Balance the Dataset
# Original data is heavily imbalanced (84% stayed, 16% left).
# To prevent the model from simply predicting "stay" every time,
# we undersample the majority class to achieve a 1:1 ratio (695:695).
# This is done without modifying the original dataset.
# ============================================================

attrition_yes = text[text['Attrition'] == 1]
attrition_no = text[text['Attrition'] == 0]

attrition_yes_sampled = attrition_yes.sample(n=695, random_state=42)
attrition_no_sampled = attrition_no.sample(n=695, random_state=42)

balanced_text = pd.concat([attrition_yes_sampled, attrition_no_sampled])
balanced_text = balanced_text.sample(frac=1, random_state=42).reset_index(drop=True)

print("\nBalanced class distribution:")
print(balanced_text['Attrition'].value_counts())

# ============================================================
# Step 4: Feature Selection
# Started with 10 theoretically relevant features, then refined
# to the 4 most impactful after iterative testing.
#
# Initial 10 features (84% accuracy on imbalanced data):
# Age, DistanceFromHome, EnvironmentSatisfaction, JobSatisfaction,
# WorkLifeBalance, MonthlyIncome, YearsAtCompany,
# YearsSinceLastPromotion, YearsWithCurrManager, PercentSalaryHike
#
# Final 4 features (67% accuracy on balanced data):
# Removing MonthlyIncome and others improved balanced accuracy.
# ============================================================

X = balanced_text[['Age', 'WorkLifeBalance', 'YearsAtCompany', 'YearsSinceLastPromotion']]
y = balanced_text['Attrition']

# ============================================================
# Step 5: Train/Test Split (70/30)
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42
)

# ============================================================
# Step 6: Train Logistic Regression Model
# liblinear solver chosen for efficiency with smaller datasets.
# ============================================================

logmodel = LogisticRegression(solver='liblinear')
logmodel.fit(X_train, y_train)

# ============================================================
# Step 7: Evaluate Model
# ============================================================

y_pred = logmodel.predict(X_test)

print('\nModel Accuracy: %0.2f' % metrics.accuracy_score(y_test, y_pred))
print('\nClassification Report:')
print(metrics.classification_report(y_test, y_pred))
print('\nConfusion Matrix:')
print(metrics.confusion_matrix(y_test, y_pred))

# ============================================================
# Step 8: Feature Coefficients
# One of the key advantages of logistic regression — each
# coefficient shows the direction and magnitude of each
# variable's influence on attrition likelihood.
# ============================================================

coef_df = pd.DataFrame({
    'Feature': X.columns,
    'Coefficient': logmodel.coef_[0]
}).sort_values('Coefficient', ascending=False)

print('\nFeature Coefficients:')
print(coef_df)
