# House Price Prediction using Machine Learning

## Week 9 – Introduction to Machine Learning Concepts

An end-to-end regression project for predicting house prices using **Area, Bedrooms, Bathrooms, Age, Location, and Property Type**.

## Project Objective

The objective of this project is to build machine learning regression models capable of predicting house prices from property characteristics.

The project demonstrates:

- Understanding and exploring the dataset
- Data cleaning and preprocessing
- Train-test splitting
- Categorical variable encoding
- Linear Regression from scratch
- Linear Regression using Scikit-learn
- Polynomial Regression
- Decision Tree Regression
- Random Forest Regression
- Model evaluation using multiple metrics
- Prediction vs actual visualization
- Feature importance analysis
- Model comparison
- Interpretation of results

---
The supplied dataset contains:

300 rows
8 columns
| Column        | Description                |
| ------------- | -------------------------- |
| Property_ID   | Unique property identifier |
| Area          | Area of the property       |
| Bedrooms      | Number of bedrooms         |
| Bathrooms     | Number of bathrooms        |
| Age           | Age of the property        |
| Location      | Location of the property   |
| Property_Type | Type of property           |
| Price         | Target house price         |

Target Variable
Price
Input Features
Area
Bedrooms
Bathrooms
Age
Location
Property_Type
Property_ID is not used for machine learning because it is only an identifier.

Note: The supplied dataset contains 300 observations. The task description does not specify a minimum dataset size for this particular project.

Project Workflow
House Price Dataset
        |
        v
Data Loading
        |
        v
Data Exploration
        |
        v
Data Cleaning
        |
        v
Feature Selection
        |
        v
Categorical Encoding
        |
        v
Train-Test Split
        |
        v
Linear Regression From Scratch
        |
        v
Scikit-Learn Linear Regression
        |
        v
Polynomial Regression
        |
        v
Decision Tree Regression
        |
        v
Random Forest Regression
        |
        v
Model Evaluation
        |
        v
Model Comparison
        |
        v
Prediction Analysis
        |
        v
Feature Importance
        |
        v
Final Insights
Exploratory Data Analysis

The following exploratory analysis was performed:

Dataset shape
Column information
Data types
Missing-value analysis
Duplicate analysis
Unique-value analysis
Descriptive statistics
Price analysis
Area analysis
Location distribution
Property-type distribution
Correlation analysis
Correlation with Price
| Feature   | Correlation |
| --------- | ----------: |
| Area      |      0.7963 |
| Bedrooms  |      0.2025 |
| Bathrooms |     -0.0303 |
| Age       |     -0.1308 |


The analysis shows that Area has the strongest numerical relationship with house Price in the supplied dataset.

Data Preparation

The following preprocessing steps were performed:

Duplicate rows were checked.
Missing values were checked and handled.
Property_ID was removed from machine learning features.
Features and target were separated.
Numerical and categorical features were identified.
Categorical variables were one-hot encoded.
The dataset was divided into training and testing sets.
Train-Test Split

The project uses:

80% Training Data
20% Testing Data

For the supplied dataset:

Total observations : 300
Training samples   : 240
Testing samples    : 60
Numerical Features
Area
Bedrooms
Bathrooms
Age
Categorical Features
Location
Property_Type
Machine Learning Models
1. Linear Regression From Scratch

Linear Regression was implemented manually using the Normal Equation:

β = (XᵀX)⁻¹Xᵀy

This demonstrates the mathematical foundation of linear regression.

2. Scikit-Learn Linear Regression

The project also implements Linear Regression using Scikit-learn:

from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)

predictions = model.predict(X_test)
3. Polynomial Regression

Polynomial Regression was implemented using polynomial features to model nonlinear relationships.

4. Decision Tree Regression

Decision Tree Regression was used to capture nonlinear relationships between property features and house prices.

5. Random Forest Regression

Random Forest Regression combines multiple decision trees and was also used to determine feature importance.

Model Evaluation

Four evaluation metrics were calculated.

Mean Absolute Error – MAE

Measures the average absolute difference between actual and predicted prices.

Lower MAE = Better Performance
Mean Squared Error – MSE

Measures the average squared prediction error.

Lower MSE = Better Performance
Root Mean Squared Error – RMSE

RMSE is the square root of MSE.

Lower RMSE = Better Performance
R² Score

R² measures the amount of variation in house prices explained by the model.

Higher R² = Better Performance
Model Results

The following results were obtained from the supplied dataset:
| Model                          |        MAE |     MSE |       RMSE |     R² |
| ------------------------------ | ---------: | ------: | ---------: | -----: |
| Linear Regression From Scratch | ₹2,188,736 | 8.45e12 | ₹2,907,633 | 0.9406 |
| Scikit-Learn Linear Regression | ₹2,188,736 | 8.45e12 | ₹2,907,633 | 0.9406 |
| Polynomial Regression          |        ~₹0 |      ~0 |        ~₹0 | 1.0000 |
| Decision Tree Regression       | ₹2,348,854 | 8.84e12 | ₹2,973,751 | 0.9379 |
| Random Forest Regression       | ₹1,643,061 | 4.84e12 | ₹2,200,837 | 0.9660 |

Important Observation About Polynomial Regression

Polynomial Regression achieved an apparently perfect:

R² = 1.0000
MAE ≈ ₹0
RMSE ≈ ₹0

This result should be interpreted carefully.

A perfect score on the supplied dataset does not necessarily mean that the model will perform perfectly on real-world house-price data. The result may be related to the structure of the supplied dataset and the way its price relationships are generated.

Additional validation such as:

Cross-validation
Independent test data
Hyperparameter tuning
Testing on real-world data

would be required before using the model for production house-price valuation.

Random Forest Performance

Random Forest produced strong practical performance:

MAE  : ₹1,643,060.78
MSE  : ₹4,843,684,174,383.45
RMSE : ₹2,200,837.15
R²   : 0.9660

The model explains approximately 96.6% of the variation in the test-set house prices according to the observed R² score.

Feature Importance

Random Forest feature importance showed that Area was the most important feature.

Feature Importance
| Feature                 | Importance |
| ----------------------- | ---------: |
| Area                    |     0.6888 |
| Location_City Center    |     0.1548 |
| Location_Rural          |     0.1004 |
| Location_Suburb         |     0.0297 |
| Bedrooms                |     0.0183 |
| Age                     |     0.0059 |
| Bathrooms               |     0.0009 |
| Property_Type_Villa     |     0.0005 |
| Property_Type_Apartment |     0.0004 |
| Property_Type_House     |     0.0003 |

Visualizations

The project generates the following visualizations.

Exploratory Data Analysis
price_distribution.png
area_vs_price.png
bedrooms_vs_price.png
location_distribution.png
property_type_distribution.png
correlation_heatmap.png
Model Evaluation
linear_regression_actual_vs_predicted.png
linear_regression_residuals.png
sklearn_actual_vs_predicted.png
sklearn_residual_analysis.png
Model Comparison
linear_regression_model_comparison.png
all_models_r2_comparison.png
all_models_rmse_comparison.png
Feature Analysis
random_forest_feature_importance.png
Final Model
best_model_actual_vs_predicted.png

The required submission visualization is:

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
│   ├── training_data_processed.csv
│   ├── testing_data_processed.csv
│   ├── preprocessing_summary.csv
│   ├── linear_regression_coefficients.csv
│   ├── linear_regression_metrics.csv
│   ├── linear_regression_predictions.csv
│   ├── sklearn_linear_regression_coefficients.csv
│   ├── sklearn_linear_regression_metrics.csv
│   ├── sklearn_linear_regression_predictions.csv
│   ├── complete_model_comparison.csv
│   ├── random_forest_feature_importance.csv
│   ├── final_model_predictions.csv
│   ├── final_model_summary.csv
│   └── business_insights.csv
│
└── visualizations/
    ├── price_distribution.png
    ├── area_vs_price.png
    ├── bedrooms_vs_price.png
    ├── location_distribution.png
    ├── property_type_distribution.png
    ├── correlation_heatmap.png
    ├── linear_regression_actual_vs_predicted.png
    ├── linear_regression_residuals.png
    ├── sklearn_actual_vs_predicted.png
    ├── sklearn_residual_analysis.png
    ├── linear_regression_model_comparison.png
    ├── all_models_r2_comparison.png
    ├── all_models_rmse_comparison.png
    ├── random_forest_feature_importance.png
    └── best_model_actual_vs_predicted.png
Installation
Prerequisites

Make sure the following are installed:

Python 3.x
Jupyter Notebook
Visual Studio Code
pip
Install Required Libraries

Run:

pip install -r requirements.txt

Or:

pip install pandas numpy matplotlib scikit-learn jupyter ipykernel
How to Run
Using Jupyter Notebook

Open:

house_price_prediction.ipynb

Select the correct Python virtual environment/kernel.

Then choose:

Run All
Using VS Code
Open the House-Price-Prediction folder.
Open house_price_prediction.ipynb.
Select the project virtual environment.
Run the notebook cells.
Verify that the output files are generated.
Using Python Script

Run from the project folder:

python house_price_prediction.py
Required Submission Files

According to the Week 9 task, the main submission files are:

house_price_prediction.ipynb
house_data.csv
model_evaluation_report.md
requirements.txt
predictions_vs_actual.png
Key Insights
1. Area is the strongest predictor

Area has the strongest numerical correlation with house price and the highest Random Forest feature importance.

2. Location is important

Location contributes significantly to the prediction of house prices.

3. Bedrooms have a smaller influence

Bedrooms have a positive relationship with price, but their influence is considerably smaller than Area.

4. Property age has a negative relationship

Older properties tend to have lower prices in the supplied dataset, although its influence is relatively small.

5. Random Forest performs strongly

Random Forest achieved:

R² = 0.9660

and produced lower MAE and RMSE than the Linear Regression and Decision Tree models in the observed experiment.

Limitations

The supplied dataset contains only 300 observations and a limited set of property features.

Real-world house-price prediction would benefit from additional variables such as:

Exact location
Neighborhood
Latitude and longitude
Distance from schools
Distance from public transportation
Parking availability
Floor number
Property condition
Construction quality
Market trends
Date of sale
Economic indicators
Larger real-world datasets

Therefore, the current project should be considered an educational machine learning project rather than a production property valuation system.

Future Enhancements

Future versions can include:

K-Fold Cross-Validation
Hyperparameter tuning
Gradient Boosting
XGBoost
Advanced feature engineering
Outlier detection
Automated model selection
Streamlit web application
FastAPI model deployment
Cloud deployment
Real-time house-price prediction
Conclusion

This project demonstrates a complete introductory machine learning workflow for house-price prediction.

The project starts with dataset exploration and cleaning and continues through feature preparation, categorical encoding, train-test splitting, model development, evaluation, visualization, and feature interpretation.

Five regression approaches were evaluated:

Linear Regression From Scratch
Scikit-Learn Linear Regression
Polynomial Regression
Decision Tree Regression
Random Forest Regression

The models were evaluated using:

MAE
MSE
RMSE
R² Score

The analysis indicates that Area is the most important predictor of house price in the supplied dataset, while Location is also an important factor.

The project demonstrates how machine learning can be used to estimate house prices and how multiple models and evaluation metrics can be compared to understand model performance.  
# Dataset

The dataset used in this project is:

```text
house_data.csv
