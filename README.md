# Student Performance Indicator — ML Model Application

## Problem Statement
This project studies how a student's performance (test scores) is affected by
other variables such as **Gender, Race/Ethnicity, Parental level of education,
Lunch type, and Test preparation course**. The goal is to build and compare
regression models that predict a student's **Math Score** from the other
available features, then select the best-performing model.

## Dataset
- **Source:** [Kaggle — Students Performance in Exams](https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977)
- **Size:** 1000 rows, 8 columns
- **Columns:** gender, race/ethnicity, parental level of education, lunch,
  test preparation course, math score, reading score, writing score

## Project Life Cycle
1. Understanding the Problem Statement
2. Data Collection
3. Data Checks (nulls, duplicates, dtypes, unique values, stats)
4. Exploratory Data Analysis (EDA)
5. Data Pre-Processing (encoding + scaling via `ColumnTransformer`)
6. Model Training (8 regression algorithms compared)
7. Choosing the Best Model

## Approach
- **Target:** Math Score
- **Features:** Reading score, writing score (scaled) + gender, ethnicity,
  parental education, lunch, test preparation course (one-hot encoded)
- **Models compared:** Linear Regression, Ridge, Lasso, K-Neighbors Regressor,
  Decision Tree, Random Forest, AdaBoost, XGBoost
- All models were evaluated on **RMSE, MAE, and R²** on both train and test
  sets to check for overfitting.

## Results

| Model | Test RMSE | Test MAE | Test R² |
|---|---|---|---|
| **Ridge Regression** | 5.39 | 4.21 | **0.880** |
| Linear Regression | 5.39 | 4.21 | 0.880 |
| Random Forest | 6.05 | 4.73 | 0.850 |
| AdaBoost | 6.16 | 4.83 | 0.844 |
| Lasso | 6.52 | 5.16 | 0.825 |
| XGBoost | 6.60 | 5.13 | 0.821 |
| K-Neighbors | 7.35 | 5.71 | 0.778 |
| Decision Tree | 8.39 | 6.67 | 0.711 |

**Best Model: Ridge Regression** — highest test R² and lowest test RMSE, with
train/test scores nearly identical (no overfitting), unlike the tree-based
models which fit the training data almost perfectly but generalized worse.

## Key Insights
- Reading and writing scores are the strongest predictors of math score.
- Completing the test preparation course is associated with higher scores.
- Students with "standard" lunch tend to score higher than "free/reduced."
- Parental level of education shows a mild positive relationship with scores.

## Files
- `Student_Performance_ML_Project.ipynb` — full notebook (EDA, preprocessing,
  model training, evaluation, and final model selection)
- `StudentsPerformance.csv` — dataset used

## How to Run
1. Clone the repo and open the notebook in Jupyter or Google Colab.
2. Ensure the CSV file is in the same directory as the notebook.
3. Run all cells top to bottom.

## Author
Hayakreev Raja — B.E. Statistics, PSG College of Arts and Science
