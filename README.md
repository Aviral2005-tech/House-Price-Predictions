House Price Prediction using Machine Learning

Week 9 – Introduction to Machine Learning Concepts

An end-to-end regression project for predicting house prices using Area, Bedrooms, Bathrooms, Age, Location, and Property Type.

Objectives

Understand and explore the house-price dataset

Clean and prepare the data

Perform an 80/20 train-test split

Encode categorical variables using One-Hot Encoding

Implement Linear Regression from scratch

Implement Linear Regression using Scikit-learn

Build Polynomial Regression, Decision Tree, and Random Forest models

Evaluate models using MAE, MSE, RMSE, and R²

Visualize predictions versus actual prices

Analyze feature importance

Compare models and identify the strongest observed model

Dataset

File: house_data.csv

The supplied dataset contains 300 rows and 8 columns.

Column

Description

Property_ID

Unique property identifier

Area

Property area

Bedrooms

Number of bedrooms

Bathrooms

Number of bathrooms

Age

Property age

Location

Property location

Property_Type

Type of property

Price

Target house price

Target: Price

Features: Area, Bedrooms, Bathrooms, Age, Location, Property_Type.

Property_ID is excluded from modeling because it is only an identifier.

The task page describes a house-price dataset but does not specify a minimum row count. The supplied dataset contains 300 observations.

Project Workflow

Dataset
  ↓
Data Loading & Exploration
  ↓
Data Cleaning
  ↓
Feature Selection
  ↓
Categorical Encoding
  ↓
80/20 Train-Test Split
  ↓
Linear Regression From Scratch
  ↓
Scikit-learn Linear Regression
  ↓
Polynomial Regression
  ↓
Decision Tree Regression
  ↓
Random Forest Regression
  ↓
MAE / MSE / RMSE / R²
  ↓
Model Comparison
  ↓
Predictions vs Actual
  ↓
Feature Importance & Insights

Exploratory Data Analysis

The project performs:

Missing-value analysis

Duplicate analysis

Descriptive statistics

Price distribution

Area vs Price analysis

Bedrooms vs Price analysis

Location distribution

Property-type distribution

Correlation analysis

Observed numerical correlations with price:

Feature

Correlation

Area

0.7963

Bedrooms

0.2025

Bathrooms

-0.0303

Age

-0.1308

Area has the strongest numerical relationship with Price in the supplied dataset.

Data Preparation

The project:

Removes duplicate rows.

Handles missing values.

Removes Property_ID from the ML features.

Separates features and target.

Uses an 80% training / 20% testing split.

One-hot encodes Location and Property_Type.

For the 300-row dataset:

Training samples: 240
Testing samples : 60

Models

1. Linear Regression From Scratch

The Normal Equation is implemented manually:

β = (XᵀX)⁻¹Xᵀy

2. Scikit-learn Linear Regression

Uses:

from sklearn.linear_model import LinearRegression

3. Polynomial Regression

Uses polynomial features to model nonlinear relationships.

4. Decision Tree Regression

Captures nonlinear relationships and feature interactions.

5. Random Forest Regression

Uses an ensemble of decision trees and also provides feature importance.

Evaluation Metrics

MAE

Average absolute prediction error. Lower is better.

MSE

Average squared prediction error. Lower is better.

RMSE

Square root of MSE. Lower is better.

R²

Measures explained variance. Higher is better.

Observed Model Results

Results from the supplied dataset and current implementation:

Model

MAE

RMSE

R²

Linear Regression From Scratch

₹2,188,736

₹2,907,633

0.9406

Scikit-learn Linear Regression

₹2,188,736

₹2,907,633

0.9406

Polynomial Regression

~₹0

~₹0

1.0000

Decision Tree

₹2,348,854

₹2,973,751

0.9379

Random Forest

₹1,643,061

₹2,200,837

0.9660

Important Note About Polynomial Regression

The supplied dataset produced an apparently perfect Polynomial Regression score. This should be interpreted cautiously. A perfect score on a small synthetic/structured dataset does not mean that the model will achieve perfect performance on real-world house prices. Additional validation such as cross-validation and testing on independent real-world data would be required.

Random Forest produced the strongest practical result among the evaluated ensemble/tree approaches:

MAE  : ₹1,643,060.78
RMSE : ₹2,200,837.15
R²   : 0.9660

Feature Importance

Random Forest identified Area as the most important feature.

Leading observed importances:

Feature

Importance

Area

0.6888

Location_City Center

0.1548

Location_Rural

0.1004

Location_Suburb

0.0297

Bedrooms

0.0183

Visualizations

The project generates:

price_distribution.png

area_vs_price.png

bedrooms_vs_price.png

location_distribution.png

property_type_distribution.png

correlation_heatmap.png

sklearn_actual_vs_predicted.png

sklearn_residual_analysis.png

all_models_r2_comparison.png

all_models_rmse_comparison.png

random_forest_feature_importance.png

best_model_actual_vs_predicted.png

The required submission image is also generated as:

predictions_vs_actual.png

Project Structure

House-Price-Prediction/
│
├── house_price_prediction.ipynb
├── house_price_prediction.py
├── house_data.csv
├── model_evaluation_report.md
├── requirements.txt
├── README.md
├── predictions_vs_actual.png
│
├── outputs/
│   ├── descriptive_statistics.csv
│   ├── correlation_matrix.csv
│   ├── dataset_summary.csv
│   ├── linear_regression_coefficients.csv
│   ├── complete_model_comparison.csv
│   ├── random_forest_feature_importance.csv
│   └── final_model_predictions.csv
│
└── visualizations/
    ├── price_distribution.png
    ├── area_vs_price.png
    ├── bedrooms_vs_price.png
    ├── location_distribution.png
    ├── property_type_distribution.png
    ├── correlation_heatmap.png
    ├── sklearn_actual_vs_predicted.png
    ├── sklearn_residual_analysis.png
    ├── all_models_r2_comparison.png
    ├── all_models_rmse_comparison.png
    ├── random_forest_feature_importance.png
    └── best_model_actual_vs_predicted.png

Installation

Prerequisites:

Python 3.x

Jupyter Notebook/JupyterLab

VS Code (recommended)

Install dependencies:

pip install -r requirements.txt

Or:

pip install pandas numpy matplotlib scikit-learn jupyter ipykernel

How to Run

Jupyter Notebook

Open:

house_price_prediction.ipynb

Select your virtual environment/kernel and choose Run All.

Python Script

From the project folder:

python house_price_prediction.py

Required Submission Files

The Week 9 task specifies:

house_price_prediction.ipynb
house_data.csv
model_evaluation_report.md
requirements.txt
predictions_vs_actual.png

Key Insights

Area is the strongest numerical predictor of price.

Location is an important categorical predictor.

Bedrooms have a positive but weaker relationship with price than area.

Property age has a comparatively small negative relationship with price.

Random Forest provides strong predictive performance in the current experiment.

Model results should be validated further before use for real-world property valuation.

Limitations

The supplied dataset contains 300 observations and a limited number of property attributes. Real-world prediction could be improved with:

Exact geographic coordinates

Neighborhood information

Distance to schools and transport

Parking

Floor number

Property condition

Construction quality

Market trends

Sale/listing date

Larger real-world datasets

Future Enhancements

K-Fold cross-validation

Hyperparameter tuning

Gradient Boosting / XGBoost

Better feature engineering

Outlier detection

Streamlit prediction application

FastAPI deployment

Cloud deployment

Real-time property-price estimation

Conclusion

This project demonstrates a complete introductory machine-learning workflow for house-price prediction. It covers data exploration, preprocessing, train-test splitting, categorical encoding, model training, evaluation, visualization, model comparison, and feature interpretation.

The analysis shows that Area and Location are particularly important predictors in the supplied dataset. The project also demonstrates why multiple models and evaluation metrics should be compared instead of relying on a single algorithm.
