# House Price Prediction - Model Evaluation Report

## 1. Project Overview

The objective of this project is to build a machine learning model capable of predicting house prices using property characteristics such as area, bedrooms, bathrooms, age, location, and property type.

## 2. Dataset

- Total rows: 300
- Total columns: 8
- Training samples: 240
- Testing samples: 60

> Note: The supplied dataset contains 300 rows. The project assignment mentions a 500+ row dataset, so the actual dataset size is reported transparently.

## 3. Models Evaluated

- Linear Regression From Scratch
- Scikit-Learn Linear Regression
- Polynomial Regression
- Decision Tree
- Random Forest

## 4. Model Performance

| Model | MAE | MSE | RMSE | R² Score |
|---|---:|---:|---:|---:|
| Linear Regression From Scratch | ₹2,188,736.34 | 8,454,330,868,424.08 | ₹2,907,633.21 | 0.9406 |
| Scikit-Learn Linear Regression | ₹2,188,736.34 | 8,454,330,868,276.58 | ₹2,907,633.21 | 0.9406 |
| Polynomial Regression | ₹0.00 | 0.00 | ₹0.00 | 1.0000 |
| Decision Tree | ₹2,348,854.24 | 8,843,197,182,745.06 | ₹2,973,751.37 | 0.9379 |
| Random Forest | ₹1,643,060.78 | 4,843,684,174,383.45 | ₹2,200,837.15 | 0.9660 |

## 5. Best Model

**Polynomial Regression** was selected as the best model based on the highest R² score.

- R² Score: 1.0000
- MAE: ₹0.00
- RMSE: ₹0.00

## 6. Feature Importance

The most important feature identified by the Random Forest model is **Area** with an importance score of **0.6888**.

## 7. Business Insights

- The best-performing model is Polynomial Regression.
- The model achieved an R² score of 1.0000.
- The model's average absolute prediction error (MAE) is ₹0.00.
- The model's RMSE is ₹0.00.
- The most important feature according to Random Forest is Area, with an importance score of 0.6888.
- Area showed the strongest linear relationship with house price during exploratory analysis.
- The trained model can be used as a baseline system for estimating house prices from property characteristics.

## 8. Conclusion

The project demonstrates a complete machine learning workflow for house price prediction, including data preparation, train-test splitting, linear regression from scratch, Scikit-Learn regression, polynomial regression, decision tree regression, and random forest regression. Model performance was evaluated using MAE, MSE, RMSE, and R² score. The selected model provides a data-driven approach for estimating property prices and identifying important pricing factors.
