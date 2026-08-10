import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_FILE = os.path.join(BASE_DIR, "house_data.csv")

OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
VISUAL_DIR = os.path.join(BASE_DIR, "visualizations")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(VISUAL_DIR, exist_ok=True)
print("=" * 70)
print("HOUSE PRICE PREDICTION - PART 1")
print("=" * 70)

print("\nCurrent Working Directory:")
print(BASE_DIR)

print("\nDataset Location:")
print(DATA_FILE)

print("\nOutput Folder:")
print(OUTPUT_DIR)

print("\nVisualization Folder:")
print(VISUAL_DIR)

print("\n" + "=" * 70)
print("1. LOADING DATASET")
print("=" * 70)

df = pd.read_csv(DATA_FILE)

print("\nDataset loaded successfully.")
print("\n" + "=" * 70)
print("2. DATASET PREVIEW")
print("=" * 70)

print("\nFirst 5 rows:")
print(df.head())

print("\nLast 5 rows:")
print(df.tail())

print("3. DATASET SHAPE")
print("=" * 70)

rows, columns = df.shape

print(f"\nNumber of Rows    : {rows}")
print(f"Number of Columns : {columns}")
print("\n" + "=" * 70)
print("4. COLUMN INFORMATION")
print("=" * 70)

print("\nColumn Names:")
print(df.columns.tolist())

print("\nData Types:")
print(df.dtypes)

print("\nDataset Information:")
df.info()

print("\n" + "=" * 70)
print("5. MISSING VALUE ANALYSIS")
print("=" * 70)

missing_values = df.isnull().sum()

print("\nMissing Values:")
print(missing_values)

total_missing = missing_values.sum()

print(f"\nTotal Missing Values: {total_missing}")

print("\n" + "=" * 70)
print("6. DUPLICATE ANALYSIS")
print("=" * 70)

duplicate_count = df.duplicated().sum()

print(f"\nNumber of Duplicate Rows: {duplicate_count}")

print("\n" + "=" * 70)
print("7. UNIQUE VALUE ANALYSIS")
print("=" * 70)

for column in df.columns:
    print(f"\n{column}:")
    print(f"Unique Values: {df[column].nunique()}")

    if df[column].dtype == "object":
        print(df[column].unique())
print("\n" + "=" * 70)
print("8. DESCRIPTIVE STATISTICS")
print("=" * 70)

statistics = df.describe()

print("\nNumerical Statistics:")
print(statistics)

statistics.to_csv(
    os.path.join(OUTPUT_DIR, "descriptive_statistics.csv")
)

print("\n" + "=" * 70)
print("9. CATEGORICAL VARIABLE ANALYSIS")
print("=" * 70)

print("\nLocation Distribution:")
print(df["Location"].value_counts())

print("\nProperty Type Distribution:")
print(df["Property_Type"].value_counts())
print("\n" + "=" * 70)
print("10. PRICE ANALYSIS")
print("=" * 70)

print(f"\nMinimum Price : {df['Price'].min():,.2f}")
print(f"Maximum Price : {df['Price'].max():,.2f}")
print(f"Average Price : {df['Price'].mean():,.2f}")
print(f"Median Price  : {df['Price'].median():,.2f}")

print("\n" + "=" * 70)
print("11. AREA ANALYSIS")
print("=" * 70)

print(f"\nMinimum Area : {df['Area'].min():,.2f}")
print(f"Maximum Area : {df['Area'].max():,.2f}")
print(f"Average Area : {df['Area'].mean():,.2f}")
print(f"Median Area  : {df['Area'].median():,.2f}")

print("\n" + "=" * 70)
print("12. CORRELATION WITH HOUSE PRICE")
print("=" * 70)

numerical_columns = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Age",
    "Price"
]

correlation_matrix = df[numerical_columns].corr()

print("\nCorrelation Matrix:")
print(correlation_matrix)

correlation_matrix.to_csv(
    os.path.join(OUTPUT_DIR, "correlation_matrix.csv")
)

price_correlation = (
    correlation_matrix["Price"]
    .sort_values(ascending=False)
)

print("\nCorrelation with Price:")
print(price_correlation)
plt.figure(figsize=(10, 6))

plt.hist(
    df["Price"],
    bins=20,
    edgecolor="black"
)

plt.title("House Price Distribution")
plt.xlabel("House Price")
plt.ylabel("Number of Properties")

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUAL_DIR,
        "price_distribution.png"
    ),
    dpi=300
)

plt.close()
plt.figure(figsize=(10, 6))

plt.scatter(
    df["Area"],
    df["Price"],
    alpha=0.7
)

plt.title("Area vs House Price")
plt.xlabel("Area")
plt.ylabel("House Price")

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUAL_DIR,
        "area_vs_price.png"
    ),
    dpi=300
)

plt.close()

plt.figure(figsize=(10, 6))

plt.scatter(
    df["Bedrooms"],
    df["Price"],
    alpha=0.7
)

plt.title("Bedrooms vs House Price")
plt.xlabel("Number of Bedrooms")
plt.ylabel("House Price")

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUAL_DIR,
        "bedrooms_vs_price.png"
    ),
    dpi=300
)

plt.close()

location_counts = df["Location"].value_counts()

plt.figure(figsize=(10, 6))

location_counts.plot(
    kind="bar"
)

plt.title("Properties by Location")
plt.xlabel("Location")
plt.ylabel("Number of Properties")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUAL_DIR,
        "location_distribution.png"
    ),
    dpi=300
)

plt.close()

property_type_counts = df["Property_Type"].value_counts()

plt.figure(figsize=(10, 6))

property_type_counts.plot(
    kind="bar"
)

plt.title("Properties by Property Type")
plt.xlabel("Property Type")
plt.ylabel("Number of Properties")

plt.xticks(rotation=45)

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUAL_DIR,
        "property_type_distribution.png"
    ),
    dpi=300
)

plt.close()
plt.figure(figsize=(10, 7))

plt.imshow(
    correlation_matrix,
    cmap="coolwarm",
    interpolation="nearest"
)

plt.colorbar()

plt.xticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns,
    rotation=45
)

plt.yticks(
    range(len(correlation_matrix.columns)),
    correlation_matrix.columns
)

plt.title("Correlation Heatmap")

plt.tight_layout()

plt.savefig(
    os.path.join(
        VISUAL_DIR,
        "correlation_heatmap.png"
    ),
    dpi=300
)

plt.close()
summary = pd.DataFrame({
    "Metric": [
        "Number of Rows",
        "Number of Columns",
        "Missing Values",
        "Duplicate Rows",
        "Average Price",
        "Median Price",
        "Minimum Price",
        "Maximum Price",
        "Average Area",
        "Median Area"
    ],
    "Value": [
        rows,
        columns,
        total_missing,
        duplicate_count,
        df["Price"].mean(),
        df["Price"].median(),
        df["Price"].min(),
        df["Price"].max(),
        df["Area"].mean(),
        df["Area"].median()
    ]
})

summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "dataset_summary.csv"
    ),
    index=False
)
print("\n" + "=" * 70)
print("PART 1 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nGenerated Files:")

print("\nOutputs:")
print("- descriptive_statistics.csv")
print("- correlation_matrix.csv")
print("- dataset_summary.csv")

print("\nVisualizations:")
print("- price_distribution.png")
print("- area_vs_price.png")
print("- bedrooms_vs_price.png")
print("- location_distribution.png")
print("- property_type_distribution.png")
print("- correlation_heatmap.png")

print("\nNext Step:")
print("Part 2 - Data Preparation & Train-Test Split")

print("=" * 70)
# ==============================================================================
# PART 2 - DATA PREPARATION & TRAIN-TEST SPLIT
# ==============================================================================

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer


print("\n" + "=" * 70)
print("HOUSE PRICE PREDICTION - PART 2")
print("=" * 70)

ml_data = df.copy()

print("\nDataset copied for machine learning preparation.")

before_duplicates = len(ml_data)

ml_data = ml_data.drop_duplicates()

after_duplicates = len(ml_data)

print("\nDuplicate Handling:")
print(f"Rows before removing duplicates : {before_duplicates}")
print(f"Rows after removing duplicates  : {after_duplicates}")
print(f"Duplicates removed              : {before_duplicates - after_duplicates}")

print("\n" + "-" * 70)
print("MISSING VALUE HANDLING")
print("-" * 70)

print("\nMissing values before processing:")
print(ml_data.isnull().sum())


# Numerical columns
numeric_features = [
    "Area",
    "Bedrooms",
    "Bathrooms",
    "Age"
]

categorical_features = [
    "Location",
    "Property_Type"
]

for column in numeric_features:

    if ml_data[column].isnull().sum() > 0:

        ml_data[column] = ml_data[column].fillna(
            ml_data[column].median()
        )


# Fill categorical missing values with mode
for column in categorical_features:

    if ml_data[column].isnull().sum() > 0:

        ml_data[column] = ml_data[column].fillna(
            ml_data[column].mode()[0]
        )


print("\nMissing values after processing:")
print(ml_data.isnull().sum())

print("\n" + "-" * 70)
print("FEATURE AND TARGET SEPARATION")
print("-" * 70)

X = ml_data.drop(
    columns=[
        "Price",
        "Property_ID"
    ]
)

y = ml_data["Price"]
print("\nFeatures:")
print(X.columns.tolist())

print("\nTarget:")
print("Price")

print("\nNumerical Features:")
print(numeric_features)

print("\nCategorical Features:")
print(categorical_features)

print("\n" + "-" * 70)
print("TRAIN-TEST SPLIT")
print("-" * 70)


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.20,
    random_state=42
)


print(f"\nTotal observations : {len(X)}")
print(f"Training samples   : {len(X_train)}")
print(f"Testing samples    : {len(X_test)}")

print("\nTraining data : 80%")
print("Testing data  : 20%")
print("\n" + "-" * 70)
print("CATEGORICAL ENCODING")
print("-" * 70)


encoder = OneHotEncoder(
    handle_unknown="ignore",
    sparse_output=False
)


X_train_encoded_categories = encoder.fit_transform(
    X_train[categorical_features]
)

X_test_encoded_categories = encoder.transform(
    X_test[categorical_features]
)

encoded_feature_names = encoder.get_feature_names_out(
    categorical_features
)


print("\nEncoded categorical features:")

for feature in encoded_feature_names:
    print("-", feature)

X_train_numeric = X_train[numeric_features].to_numpy()

X_test_numeric = X_test[numeric_features].to_numpy()

X_train_processed = np.hstack(
    (
        X_train_numeric,
        X_train_encoded_categories
    )
)

X_test_processed = np.hstack(
    (
        X_test_numeric,
        X_test_encoded_categories
    )
)
# Final feature names
processed_feature_names = (
    numeric_features
    + list(encoded_feature_names)
)


print("\n" + "-" * 70)
print("PROCESSED DATA")
print("-" * 70)

print(
    f"\nTraining data shape : "
    f"{X_train_processed.shape}"
)

print(
    f"Testing data shape  : "
    f"{X_test_processed.shape}"
)

print(
    f"Total ML features   : "
    f"{len(processed_feature_names)}"
)

X_train_processed_df = pd.DataFrame(
    X_train_processed,
    columns=processed_feature_names
)

X_test_processed_df = pd.DataFrame(
    X_test_processed,
    columns=processed_feature_names
)


# Add target column
X_train_processed_df["Price"] = y_train.reset_index(
    drop=True
)

X_test_processed_df["Price"] = y_test.reset_index(
    drop=True
)

X_train_processed_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "training_data_processed.csv"
    ),
    index=False
)

X_test_processed_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "testing_data_processed.csv"
    ),
    index=False
)

preprocessing_summary = pd.DataFrame({

    "Metric": [
        "Original Rows",
        "Rows After Duplicate Removal",
        "Training Samples",
        "Testing Samples",
        "Training Percentage",
        "Testing Percentage",
        "Numerical Features",
        "Categorical Features",
        "Encoded Categorical Features",
        "Total Processed Features"
    ],

    "Value": [
        len(df),
        len(ml_data),
        len(X_train),
        len(X_test),
        "80%",
        "20%",
        len(numeric_features),
        len(categorical_features),
        len(encoded_feature_names),
        len(processed_feature_names)
    ]
})


preprocessing_summary.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "preprocessing_summary.csv"
    ),
    index=False
)


# ==============================================================================
# 14. DISPLAY FINAL FEATURE TABLE
# ==============================================================================

print("\n" + "-" * 70)
print("FINAL FEATURE LIST")
print("-" * 70)

for number, feature in enumerate(
    processed_feature_names,
    start=1
):

    print(f"{number}. {feature}")

print("\n" + "=" * 70)
print("PART 2 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nCompleted Tasks:")
print("✓ Duplicate checking")
print("✓ Missing value handling")
print("✓ Property_ID removed")
print("✓ Features and target separated")
print("✓ Categorical variables encoded")
print("✓ 80/20 train-test split")
print("✓ Processed training data saved")
print("✓ Processed testing data saved")
print("✓ Preprocessing summary saved")

print("\nGenerated Files:")
print("- training_data_processed.csv")
print("- testing_data_processed.csv")
print("- preprocessing_summary.csv")

print("\nNext Step:")
print("Part 3 - Linear Regression From Scratch")

print("=" * 70)

print("\n" + "=" * 70)
print("HOUSE PRICE PREDICTION - PART 3")
print("LINEAR REGRESSION FROM SCRATCH")
print("=" * 70)
print("\n1. Preparing data for Linear Regression...")


# Add a column of 1s for the intercept
X_train_scratch = np.column_stack(
    (
        np.ones(X_train_processed.shape[0]),
        X_train_processed
    )
)

X_test_scratch = np.column_stack(
    (
        np.ones(X_test_processed.shape[0]),
        X_test_processed
    )
)


# Convert target values to NumPy arrays
y_train_array = y_train.to_numpy()
y_test_array = y_test.to_numpy()


print("Training matrix shape:", X_train_scratch.shape)
print("Testing matrix shape :", X_test_scratch.shape)

print("\n2. Calculating Linear Regression coefficients...")


# Linear Regression Normal Equation:
#
# β = (XᵀX)^(-1) Xᵀy
#
# To improve numerical stability, pseudoinverse is used.

XTX = X_train_scratch.T @ X_train_scratch

XTX_inverse = np.linalg.pinv(XTX)

XTy = X_train_scratch.T @ y_train_array

beta = XTX_inverse @ XTy

intercept = beta[0]

coefficients = beta[1:]


print("\nIntercept:")
print(intercept)


print("\nNumber of coefficients:")
print(len(coefficients))

print("\n" + "-" * 70)
print("MODEL COEFFICIENTS")
print("-" * 70)


for feature, coefficient in zip(
    processed_feature_names,
    coefficients
):

    print(
        f"{feature:<35} : {coefficient:,.2f}"
    )

print("\n" + "-" * 70)
print("MAKING PREDICTIONS")
print("-" * 70)


y_pred_scratch = (
    X_test_scratch @ beta
)


print(
    "\nPredictions generated successfully."
)

print(
    "Number of predictions:",
    len(y_pred_scratch)
)

mae_scratch = np.mean(
    np.abs(
        y_test_array - y_pred_scratch
    )
)

mse_scratch = np.mean(
    (
        y_test_array - y_pred_scratch
    ) ** 2
)

rmse_scratch = np.sqrt(
    mse_scratch
)

ss_res = np.sum(
    (
        y_test_array - y_pred_scratch
    ) ** 2
)


ss_tot = np.sum(
    (
        y_test_array -
        np.mean(y_test_array)
    ) ** 2
)


r2_scratch = 1 - (
    ss_res / ss_tot
)

print("\n" + "=" * 70)
print("LINEAR REGRESSION FROM SCRATCH - PERFORMANCE")
print("=" * 70)


print(
    f"\nMAE  : ₹{mae_scratch:,.2f}"
)

print(
    f"MSE  : {mse_scratch:,.2f}"
)

print(
    f"RMSE : ₹{rmse_scratch:,.2f}"
)

print(
    f"R²   : {r2_scratch:.4f}"
)

coefficients_df = pd.DataFrame({

    "Feature": processed_feature_names,

    "Coefficient": coefficients

})


coefficients_df.loc[
    len(coefficients_df)
] = [
    "Intercept",
    intercept
]


coefficients_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "linear_regression_coefficients.csv"
    ),

    index=False
)

model_metrics = pd.DataFrame({

    "Metric": [
        "MAE",
        "MSE",
        "RMSE",
        "R2 Score"
    ],

    "Value": [
        mae_scratch,
        mse_scratch,
        rmse_scratch,
        r2_scratch
    ]

})


model_metrics.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "linear_regression_metrics.csv"
    ),

    index=False
)

predictions_df = pd.DataFrame({

    "Actual_Price": y_test_array,

    "Predicted_Price": y_pred_scratch

})


predictions_df["Absolute_Error"] = (

    np.abs(
        predictions_df["Actual_Price"]
        -
        predictions_df["Predicted_Price"]
    )

)


predictions_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "linear_regression_predictions.csv"
    ),

    index=False
)

plt.figure(
    figsize=(10, 7)
)


plt.scatter(
    y_test_array,
    y_pred_scratch,
    alpha=0.7
)


# Perfect prediction line
minimum = min(
    y_test_array.min(),
    y_pred_scratch.min()
)

maximum = max(
    y_test_array.max(),
    y_pred_scratch.max()
)


plt.plot(
    [minimum, maximum],
    [minimum, maximum],
    linestyle="--"
)


plt.xlabel(
    "Actual House Price"
)

plt.ylabel(
    "Predicted House Price"
)

plt.title(
    "Linear Regression - Actual vs Predicted Prices"
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        VISUAL_DIR,
        "linear_regression_actual_vs_predicted.png"
    ),

    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "\nVisualization saved:"
)

print(
    "- linear_regression_actual_vs_predicted.png"
)

residuals = (
    y_test_array -
    y_pred_scratch
)


plt.figure(
    figsize=(10, 7)
)


plt.scatter(
    y_pred_scratch,
    residuals,
    alpha=0.7
)


plt.axhline(
    y=0,
    linestyle="--"
)


plt.xlabel(
    "Predicted House Price"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Linear Regression - Residual Analysis"
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        VISUAL_DIR,
        "linear_regression_residuals.png"
    ),

    dpi=300,
    bbox_inches="tight"
)


plt.close()


print(
    "- linear_regression_residuals.png"
)


# ==============================================================================
# 16. PART 3 COMPLETION
# ==============================================================================

print("\n" + "=" * 70)
print("PART 3 COMPLETED SUCCESSFULLY")
print("=" * 70)


print("\nLinear Regression From Scratch:")
print("✓ Normal Equation implemented")
print("✓ Model coefficients calculated")
print("✓ Intercept calculated")
print("✓ Test predictions generated")
print("✓ MAE calculated")
print("✓ MSE calculated")
print("✓ RMSE calculated")
print("✓ R² Score calculated")
print("✓ Prediction results saved")
print("✓ Model coefficients saved")
print("✓ Model metrics saved")
print("✓ Actual vs predicted visualization created")
print("✓ Residual visualization created")


print("\nGenerated Files:")

print("\nOutputs:")
print("- linear_regression_coefficients.csv")
print("- linear_regression_metrics.csv")
print("- linear_regression_predictions.csv")

print("\nVisualizations:")
print("- linear_regression_actual_vs_predicted.png")
print("- linear_regression_residuals.png")


print("\nNext Step:")
print("Part 4 - Linear Regression Using Scikit-Learn")


print("=" * 70)
# ==============================================================================
# PART 4 - LINEAR REGRESSION USING SCIKIT-LEARN
# ==============================================================================

from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


print("\n" + "=" * 70)
print("HOUSE PRICE PREDICTION - PART 4")
print("LINEAR REGRESSION USING SCIKIT-LEARN")
print("=" * 70)
print("\n1. Creating Linear Regression model...")

sklearn_model = LinearRegression()

print("Model created successfully.")
print("\n2. Training the model...")

sklearn_model.fit(
    X_train_processed,
    y_train
)

print("Model training completed successfully.")
print("\n3. Generating predictions...")

y_pred_sklearn = sklearn_model.predict(
    X_test_processed
)

print(
    f"Number of predictions: {len(y_pred_sklearn)}"
)
sklearn_mae = mean_absolute_error(
    y_test,
    y_pred_sklearn
)

sklearn_mse = mean_squared_error(
    y_test,
    y_pred_sklearn
)

sklearn_rmse = np.sqrt(
    sklearn_mse
)

sklearn_r2 = r2_score(
    y_test,
    y_pred_sklearn
)

print("\n" + "=" * 70)
print("SCIKIT-LEARN LINEAR REGRESSION - PERFORMANCE")
print("=" * 70)

print(
    f"\nMAE  : ₹{sklearn_mae:,.2f}"
)

print(
    f"MSE  : {sklearn_mse:,.2f}"
)

print(
    f"RMSE : ₹{sklearn_rmse:,.2f}"
)

print(
    f"R²   : {sklearn_r2:.4f}"
)

print("\n" + "-" * 70)
print("MODEL INTERCEPT")
print("-" * 70)

print(
    f"\nIntercept: ₹{sklearn_model.intercept_:,.2f}"
)

print("\n" + "-" * 70)
print("MODEL COEFFICIENTS")
print("-" * 70)


for feature, coefficient in zip(
    processed_feature_names,
    sklearn_model.coef_
):

    print(
        f"{feature:<35} : ₹{coefficient:,.2f}"
    )

sklearn_coefficients_df = pd.DataFrame({

    "Feature": processed_feature_names,

    "Coefficient": sklearn_model.coef_

})


sklearn_coefficients_df.loc[
    len(sklearn_coefficients_df)
] = [
    "Intercept",
    sklearn_model.intercept_
]


sklearn_coefficients_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "sklearn_linear_regression_coefficients.csv"
    ),

    index=False
)

sklearn_metrics_df = pd.DataFrame({

    "Metric": [
        "MAE",
        "MSE",
        "RMSE",
        "R2 Score"
    ],

    "Value": [
        sklearn_mae,
        sklearn_mse,
        sklearn_rmse,
        sklearn_r2
    ]

})


sklearn_metrics_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "sklearn_linear_regression_metrics.csv"
    ),

    index=False
)

sklearn_predictions_df = pd.DataFrame({

    "Actual_Price": y_test.to_numpy(),

    "Predicted_Price": y_pred_sklearn

})
sklearn_predictions_df["Absolute_Error"] = (

    np.abs(
        sklearn_predictions_df["Actual_Price"]
        -
        sklearn_predictions_df["Predicted_Price"]
    )

)
sklearn_predictions_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "sklearn_linear_regression_predictions.csv"
    ),

    index=False
)

plt.figure(
    figsize=(10, 7)
)


plt.scatter(

    y_test,

    y_pred_sklearn,

    alpha=0.7
)


minimum = min(
    y_test.min(),
    y_pred_sklearn.min()
)

maximum = max(
    y_test.max(),
    y_pred_sklearn.max()
)


plt.plot(

    [minimum, maximum],

    [minimum, maximum],

    linestyle="--"
)


plt.xlabel(
    "Actual House Price"
)

plt.ylabel(
    "Predicted House Price"
)

plt.title(
    "Scikit-Learn Linear Regression - Actual vs Predicted"
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        VISUAL_DIR,
        "sklearn_actual_vs_predicted.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print(
    "\nVisualization saved:"
)

print(
    "- sklearn_actual_vs_predicted.png"
)

sklearn_residuals = (

    y_test.to_numpy()
    -
    y_pred_sklearn

)


plt.figure(
    figsize=(10, 7)
)


plt.scatter(

    y_pred_sklearn,

    sklearn_residuals,

    alpha=0.7
)


plt.axhline(

    y=0,

    linestyle="--"

)


plt.xlabel(
    "Predicted House Price"
)

plt.ylabel(
    "Residual"
)

plt.title(
    "Scikit-Learn Linear Regression - Residual Analysis"
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        VISUAL_DIR,
        "sklearn_residual_analysis.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print(
    "- sklearn_residual_analysis.png"
)
print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)


comparison_df = pd.DataFrame({

    "Model": [
        "Linear Regression From Scratch",
        "Scikit-Learn Linear Regression"
    ],

    "MAE": [
        mae_scratch,
        sklearn_mae
    ],

    "MSE": [
        mse_scratch,
        sklearn_mse
    ],

    "RMSE": [
        rmse_scratch,
        sklearn_rmse
    ],

    "R2_Score": [
        r2_scratch,
        sklearn_r2
    ]

})


print("\n")
print(
    comparison_df.to_string(
        index=False
    )
)

comparison_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "linear_regression_model_comparison.csv"
    ),

    index=False
)

if sklearn_r2 >= r2_scratch:

    better_model = (
        "Scikit-Learn Linear Regression"
    )

else:

    better_model = (
        "Linear Regression From Scratch"
    )


print(
    f"\nBetter model based on R²: "
    f"{better_model}"
)

models = [
    "From Scratch",
    "Scikit-Learn"
]

r2_values = [
    r2_scratch,
    sklearn_r2
]


plt.figure(
    figsize=(9, 6)
)


plt.bar(
    models,
    r2_values
)


plt.ylabel(
    "R² Score"
)

plt.xlabel(
    "Model"
)

plt.title(
    "Linear Regression Model Comparison"
)


plt.ylim(
    min(0, min(r2_values) - 0.05),
    min(1, max(r2_values) + 0.10)
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        VISUAL_DIR,
        "linear_regression_model_comparison.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print(
    "\nComparison chart saved:"
)

print(
    "- linear_regression_model_comparison.png"
)

print("\n" + "=" * 70)
print("PART 4 COMPLETED SUCCESSFULLY")
print("=" * 70)


print("\nCompleted Tasks:")

print("✓ Scikit-Learn Linear Regression created")
print("✓ Model trained")
print("✓ Test predictions generated")
print("✓ MAE calculated")
print("✓ MSE calculated")
print("✓ RMSE calculated")
print("✓ R² Score calculated")
print("✓ Model coefficients extracted")
print("✓ Predictions saved")
print("✓ Model metrics saved")
print("✓ Actual vs predicted visualization created")
print("✓ Residual visualization created")
print("✓ From-scratch model comparison completed")


print("\nGenerated Files:")

print("\nOutputs:")

print("- sklearn_linear_regression_coefficients.csv")
print("- sklearn_linear_regression_metrics.csv")
print("- sklearn_linear_regression_predictions.csv")
print("- linear_regression_model_comparison.csv")


print("\nVisualizations:")

print("- sklearn_actual_vs_predicted.png")
print("- sklearn_residual_analysis.png")
print("- linear_regression_model_comparison.png")


print("\nNext Step:")
print("Part 5 - Model Improvement using Polynomial Regression,")
print("Decision Tree and Random Forest")


print("=" * 70)
# ==============================================================================
# PART 5 - MODEL IMPROVEMENT & FINAL MODEL SELECTION
# ==============================================================================

from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline


print("\n" + "=" * 70)
print("HOUSE PRICE PREDICTION - PART 5")
print("MODEL IMPROVEMENT & FINAL MODEL SELECTION")
print("=" * 70)

print("\n" + "-" * 70)
print("1. POLYNOMIAL REGRESSION")
print("-" * 70)


# Polynomial features
polynomial_model = Pipeline([
    (
        "polynomial_features",
        PolynomialFeatures(
            degree=2,
            include_bias=False
        )
    ),

    (
        "linear_regression",
        LinearRegression()
    )
])


print("\nTraining Polynomial Regression...")


polynomial_model.fit(
    X_train_processed,
    y_train
)


# Predictions
y_pred_polynomial = polynomial_model.predict(
    X_test_processed
)


# Evaluation
polynomial_mae = mean_absolute_error(
    y_test,
    y_pred_polynomial
)

polynomial_mse = mean_squared_error(
    y_test,
    y_pred_polynomial
)

polynomial_rmse = np.sqrt(
    polynomial_mse
)

polynomial_r2 = r2_score(
    y_test,
    y_pred_polynomial
)


print("\nPolynomial Regression Performance:")

print(
    f"MAE  : ₹{polynomial_mae:,.2f}"
)

print(
    f"MSE  : {polynomial_mse:,.2f}"
)

print(
    f"RMSE : ₹{polynomial_rmse:,.2f}"
)

print(
    f"R²   : {polynomial_r2:.4f}"
)

print("\n" + "-" * 70)
print("2. DECISION TREE REGRESSION")
print("-" * 70)


decision_tree_model = DecisionTreeRegressor(
    max_depth=6,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42
)


print("\nTraining Decision Tree...")


decision_tree_model.fit(
    X_train_processed,
    y_train
)


# Predictions
y_pred_tree = decision_tree_model.predict(
    X_test_processed
)


# Evaluation
tree_mae = mean_absolute_error(
    y_test,
    y_pred_tree
)

tree_mse = mean_squared_error(
    y_test,
    y_pred_tree
)

tree_rmse = np.sqrt(
    tree_mse
)

tree_r2 = r2_score(
    y_test,
    y_pred_tree
)


print("\nDecision Tree Performance:")

print(
    f"MAE  : ₹{tree_mae:,.2f}"
)

print(
    f"MSE  : {tree_mse:,.2f}"
)

print(
    f"RMSE : ₹{tree_rmse:,.2f}"
)

print(
    f"R²   : {tree_r2:.4f}"
)

print("\n" + "-" * 70)
print("3. RANDOM FOREST REGRESSION")
print("-" * 70)


random_forest_model = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    min_samples_split=5,
    min_samples_leaf=2,
    random_state=42,
    n_jobs=-1
)

print("\nTraining Random Forest...")
random_forest_model.fit(
    X_train_processed,
    y_train
)


# Predictions
y_pred_forest = random_forest_model.predict(
    X_test_processed
)


# Evaluation
forest_mae = mean_absolute_error(
    y_test,
    y_pred_forest
)

forest_mse = mean_squared_error(
    y_test,
    y_pred_forest
)

forest_rmse = np.sqrt(
    forest_mse
)

forest_r2 = r2_score(
    y_test,
    y_pred_forest
)


print("\nRandom Forest Performance:")

print(
    f"MAE  : ₹{forest_mae:,.2f}"
)

print(
    f"MSE  : {forest_mse:,.2f}"
)

print(
    f"RMSE : ₹{forest_rmse:,.2f}"
)

print(
    f"R²   : {forest_r2:.4f}"
)

print("\n" + "=" * 70)
print("COMPLETE MODEL COMPARISON")
print("=" * 70)


all_models = pd.DataFrame({

    "Model": [
        "Linear Regression From Scratch",
        "Scikit-Learn Linear Regression",
        "Polynomial Regression",
        "Decision Tree",
        "Random Forest"
    ],

    "MAE": [
        mae_scratch,
        sklearn_mae,
        polynomial_mae,
        tree_mae,
        forest_mae
    ],

    "MSE": [
        mse_scratch,
        sklearn_mse,
        polynomial_mse,
        tree_mse,
        forest_mse
    ],

    "RMSE": [
        rmse_scratch,
        sklearn_rmse,
        polynomial_rmse,
        tree_rmse,
        forest_rmse
    ],

    "R2_Score": [
        r2_scratch,
        sklearn_r2,
        polynomial_r2,
        tree_r2,
        forest_r2
    ]

})


print("\n")

print(
    all_models.to_string(
        index=False
    )
)

all_models_rounded = all_models.copy()

all_models_rounded["MAE"] = (
    all_models_rounded["MAE"].round(2)
)

all_models_rounded["MSE"] = (
    all_models_rounded["MSE"].round(2)
)

all_models_rounded["RMSE"] = (
    all_models_rounded["RMSE"].round(2)
)

all_models_rounded["R2_Score"] = (
    all_models_rounded["R2_Score"].round(4)
)

all_models_rounded.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "complete_model_comparison.csv"
    ),

    index=False
)

best_model_row = all_models.loc[
    all_models["R2_Score"].idxmax()
]


best_model_name = best_model_row["Model"]

best_r2 = best_model_row["R2_Score"]

best_mae = best_model_row["MAE"]

best_rmse = best_model_row["RMSE"]


print("\n" + "=" * 70)
print("BEST MODEL")
print("=" * 70)
print(
    f"\nBest Model : {best_model_name}"
)

print(
    f"R² Score   : {best_r2:.4f}"
)

print(
    f"MAE        : ₹{best_mae:,.2f}"
)

print(
    f"RMSE       : ₹{best_rmse:,.2f}"
)

plt.figure(
    figsize=(11, 7)
)


plt.bar(
    all_models["Model"],
    all_models["R2_Score"]
)


plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "R² Score"
)

plt.title(
    "Comparison of House Price Prediction Models"
)


plt.xticks(
    rotation=30,
    ha="right"
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        VISUAL_DIR,
        "all_models_r2_comparison.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print(
    "\nVisualization saved:"
)

print(
    "- all_models_r2_comparison.png"
)
plt.figure(
    figsize=(11, 7)
)


plt.bar(
    all_models["Model"],
    all_models["RMSE"]
)


plt.xlabel(
    "Machine Learning Model"
)

plt.ylabel(
    "RMSE"
)

plt.title(
    "RMSE Comparison of House Price Prediction Models"
)


plt.xticks(
    rotation=30,
    ha="right"
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        VISUAL_DIR,
        "all_models_rmse_comparison.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print(
    "- all_models_rmse_comparison.png"
)

print("\n" + "=" * 70)
print("RANDOM FOREST FEATURE IMPORTANCE")
print("=" * 70)


feature_importance_df = pd.DataFrame({

    "Feature": processed_feature_names,

    "Importance": random_forest_model.feature_importances_

})


feature_importance_df = (
    feature_importance_df
    .sort_values(
        by="Importance",
        ascending=False
    )
)


print("\nFeature Importance:")

print(
    feature_importance_df.to_string(
        index=False
    )
)

feature_importance_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "random_forest_feature_importance.csv"
    ),

    index=False
)

plt.figure(
    figsize=(11, 7)
)


top_features = feature_importance_df.head(10)


plt.barh(
    top_features["Feature"][::-1],
    top_features["Importance"][::-1]
)


plt.xlabel(
    "Importance"
)

plt.ylabel(
    "Feature"
)

plt.title(
    "Top 10 Features - Random Forest"
)


plt.tight_layout()


plt.savefig(

    os.path.join(
        VISUAL_DIR,
        "random_forest_feature_importance.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()


print(
    "\nVisualization saved:"
)

print(
    "- random_forest_feature_importance.png"
)

model_predictions = {

    "Linear Regression From Scratch":
        y_pred_scratch,

    "Scikit-Learn Linear Regression":
        y_pred_sklearn,

    "Polynomial Regression":
        y_pred_polynomial,

    "Decision Tree":
        y_pred_tree,

    "Random Forest":
        y_pred_forest

}


best_predictions = model_predictions[
    best_model_name
]


final_predictions_df = pd.DataFrame({

    "Actual_Price":
        y_test.to_numpy(),

    "Predicted_Price":
        best_predictions

})


final_predictions_df["Absolute_Error"] = (

    np.abs(
        final_predictions_df["Actual_Price"]
        -
        final_predictions_df["Predicted_Price"]
    )

)


final_predictions_df["Percentage_Error"] = (

    final_predictions_df["Absolute_Error"]
    /
    final_predictions_df["Actual_Price"]
    *
    100

)

final_predictions_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "final_model_predictions.csv"
    ),

    index=False
)

plt.figure(
    figsize=(10, 7)
)


plt.scatter(

    y_test,

    best_predictions,

    alpha=0.7
)


minimum = min(
    y_test.min(),
    best_predictions.min()
)

maximum = max(
    y_test.max(),
    best_predictions.max()
)


plt.plot(

    [minimum, maximum],

    [minimum, maximum],

    linestyle="--"

)


plt.xlabel(
    "Actual House Price"
)

plt.ylabel(
    "Predicted House Price"
)

plt.title(
    f"{best_model_name} - Actual vs Predicted"
)
plt.tight_layout()

plt.savefig(

    os.path.join(
        VISUAL_DIR,
        "best_model_actual_vs_predicted.png"
    ),

    dpi=300,

    bbox_inches="tight"
)


plt.close()

final_model_summary = pd.DataFrame({

    "Metric": [
        "Best Model",
        "R2 Score",
        "MAE",
        "RMSE",
        "Training Samples",
        "Testing Samples",
        "Total Dataset Rows"
    ],

    "Value": [
        best_model_name,
        best_r2,
        best_mae,
        best_rmse,
        len(X_train),
        len(X_test),
        len(df)
    ]

})
final_model_summary.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "final_model_summary.csv"
    ),

    index=False
)

top_feature = (
    feature_importance_df.iloc[0]["Feature"]
)


top_feature_importance = (
    feature_importance_df.iloc[0]["Importance"]
)


business_insights = [

    f"The best-performing model is {best_model_name}.",

    f"The model achieved an R² score of {best_r2:.4f}.",

    f"The model's average absolute prediction error "
    f"(MAE) is ₹{best_mae:,.2f}.",

    f"The model's RMSE is ₹{best_rmse:,.2f}.",

    f"The most important feature according to Random Forest "
    f"is {top_feature}, with an importance score of "
    f"{top_feature_importance:.4f}.",

    "Area showed the strongest linear relationship with "
    "house price during exploratory analysis.",

    "The trained model can be used as a baseline system "
    "for estimating house prices from property characteristics."

]


business_insights_df = pd.DataFrame({

    "Business_Insight": business_insights

})


business_insights_df.to_csv(

    os.path.join(
        OUTPUT_DIR,
        "business_insights.csv"
    ),

    index=False
)
report_path = os.path.join(
    BASE_DIR,
    "model_evaluation_report.md"
)


with open(
    report_path,
    "w",
    encoding="utf-8"
) as report:

    report.write(
        "# House Price Prediction - Model Evaluation Report\n\n"
    )

    report.write(
        "## 1. Project Overview\n\n"
    )

    report.write(
        "The objective of this project is to build a machine "
        "learning model capable of predicting house prices "
        "using property characteristics such as area, bedrooms, "
        "bathrooms, age, location, and property type.\n\n"
    )

    report.write(
        "## 2. Dataset\n\n"
    )

    report.write(
        f"- Total rows: {len(df)}\n"
    )

    report.write(
        f"- Total columns: {len(df.columns)}\n"
    )

    report.write(
        f"- Training samples: {len(X_train)}\n"
    )

    report.write(
        f"- Testing samples: {len(X_test)}\n\n"
    )

    report.write(
        "> Note: The supplied dataset contains 300 rows. "
        "The project assignment mentions a 500+ row dataset, "
        "so the actual dataset size is reported transparently.\n\n"
    )

    report.write(
        "## 3. Models Evaluated\n\n"
    )

    for model_name in all_models["Model"]:
        report.write(
            f"- {model_name}\n"
        )

    report.write("\n")

    report.write(
        "## 4. Model Performance\n\n"
    )

    report.write(
        "| Model | MAE | MSE | RMSE | R² Score |\n"
    )

    report.write(
        "|---|---:|---:|---:|---:|\n"
    )

    for _, row in all_models.iterrows():

        report.write(
            f"| {row['Model']} | "
            f"₹{row['MAE']:,.2f} | "
            f"{row['MSE']:,.2f} | "
            f"₹{row['RMSE']:,.2f} | "
            f"{row['R2_Score']:.4f} |\n"
        )

    report.write("\n")

    report.write(
        "## 5. Best Model\n\n"
    )

    report.write(
        f"**{best_model_name}** was selected as the best "
        f"model based on the highest R² score.\n\n"
    )

    report.write(
        f"- R² Score: {best_r2:.4f}\n"
    )

    report.write(
        f"- MAE: ₹{best_mae:,.2f}\n"
    )

    report.write(
        f"- RMSE: ₹{best_rmse:,.2f}\n\n"
    )

    report.write(
        "## 6. Feature Importance\n\n"
    )

    report.write(
        f"The most important feature identified by the "
        f"Random Forest model is **{top_feature}** "
        f"with an importance score of "
        f"**{top_feature_importance:.4f}**.\n\n"
    )

    report.write(
        "## 7. Business Insights\n\n"
    )

    for insight in business_insights:

        report.write(
            f"- {insight}\n"
        )

    report.write("\n")

    report.write(
        "## 8. Conclusion\n\n"
    )

    report.write(
        "The project demonstrates a complete machine learning "
        "workflow for house price prediction, including data "
        "preparation, train-test splitting, linear regression "
        "from scratch, Scikit-Learn regression, polynomial "
        "regression, decision tree regression, and random "
        "forest regression. Model performance was evaluated "
        "using MAE, MSE, RMSE, and R² score. The selected model "
        "provides a data-driven approach for estimating property "
        "prices and identifying important pricing factors.\n"
    )


print(
    f"\nModel evaluation report created:\n{report_path}"
)


# ==============================================================================
# 19. FINAL COMPLETION
# ==============================================================================

print("\n" + "=" * 70)
print("PART 5 COMPLETED SUCCESSFULLY")
print("=" * 70)

print("\nModels evaluated:")
print("✓ Linear Regression From Scratch")
print("✓ Scikit-Learn Linear Regression")
print("✓ Polynomial Regression")
print("✓ Decision Tree Regression")
print("✓ Random Forest Regression")

print("\nEvaluation metrics:")
print("✓ MAE")
print("✓ MSE")
print("✓ RMSE")
print("✓ R² Score")

print("\nAdditional analysis:")
print("✓ Model comparison")
print("✓ Random Forest feature importance")
print("✓ Best model selection")
print("✓ Final predictions")
print("✓ Business insights")
print("✓ Model evaluation report")

print("\nBest Model:")
print(best_model_name)

print(f"\nBest R² Score: {best_r2:.4f}")

print("\nGenerated output files:")
print("- complete_model_comparison.csv")
print("- random_forest_feature_importance.csv")
print("- final_model_predictions.csv")
print("- final_model_summary.csv")
print("- business_insights.csv")

print("\nGenerated visualizations:")
print("- all_models_r2_comparison.png")
print("- all_models_rmse_comparison.png")
print("- random_forest_feature_importance.png")
print("- best_model_actual_vs_predicted.png")

print("\nGenerated report:")
print("- model_evaluation_report.md")

print("\n" + "=" * 70)