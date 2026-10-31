
# ============================================================
# CODEALPHA INTERNSHIP
# PROJECT: CAR PRICE PREDICTION WITH MACHINE LEARNING
# ============================================================

# 1. IMPORT LIBRARIES

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

print("Libraries imported successfully!")

sns.set_theme(style="whitegrid")


# 2. LOAD DATASET

folder = os.path.dirname(os.path.abspath(__file__))

file_path = os.path.join(folder, "C:/Users/HP/Downloads/archive (2)/car data.csv")

df = pd.read_csv(file_path)

print("\n========== DATASET INFORMATION ==========")

print(df.head())

print("\nDataset Shape:", df.shape)

print("\nColumn Names:")
print(df.columns.tolist())


# 3. DATA CLEANING

df.columns = df.columns.str.strip()

# Remove duplicate records
df.drop_duplicates(inplace=True)

# Convert numerical columns
numeric_columns = [
    "Year",
    "Selling_Price",
    "Present_Price",
    "Driven_kms",
    "Owner"
]

for col in numeric_columns:
    df[col] = pd.to_numeric(df[col], errors="coerce")

# Convert categorical columns
categorical_columns = [
    "Car_Name",
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]

for col in categorical_columns:
    df[col] = df[col].astype("string").str.strip()

# Remove records without target value
df.dropna(subset=["Selling_Price"], inplace=True)

print("\n========== DATA CLEANING ==========")

print("\nMissing Values:")
print(df.isnull().sum())

print("\nCleaned Dataset Shape:", df.shape)


# 4. EXPLORATORY DATA ANALYSIS

print("\n========== STATISTICAL SUMMARY ==========")

print(df.describe())

# Distribution of selling prices

plt.figure(figsize=(8, 5))

sns.histplot(
    df["Selling_Price"],
    kde=True
)

plt.title("Distribution of Car Selling Prices")
plt.xlabel("Selling Price")
plt.ylabel("Frequency")

plt.tight_layout()
plt.show()


# 5. FUEL TYPE ANALYSIS

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Fuel_Type",
    y="Selling_Price"
)

plt.title("Car Selling Price by Fuel Type")
plt.xlabel("Fuel Type")
plt.ylabel("Selling Price")

plt.tight_layout()
plt.show()


# 6. TRANSMISSION ANALYSIS

plt.figure(figsize=(8, 5))

sns.boxplot(
    data=df,
    x="Transmission",
    y="Selling_Price"
)

plt.title("Selling Price by Transmission Type")
plt.xlabel("Transmission")
plt.ylabel("Selling Price")

plt.tight_layout()
plt.show()


# 7. YEAR VS SELLING PRICE

plt.figure(figsize=(9, 5))

sns.scatterplot(
    data=df,
    x="Year",
    y="Selling_Price",
    hue="Fuel_Type"
)

plt.title("Car Manufacturing Year vs Selling Price")
plt.xlabel("Manufacturing Year")
plt.ylabel("Selling Price")

plt.tight_layout()
plt.show()


# 8. CORRELATION ANALYSIS

plt.figure(figsize=(8, 6))

correlation = df[
    numeric_columns
].corr()

sns.heatmap(
    correlation,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)

plt.title("Correlation Between Numerical Features")

plt.tight_layout()
plt.show()


# 9. FEATURE SELECTION

X = df.drop("Selling_Price", axis=1)

y = df["Selling_Price"]

numeric_features = [
    "Year",
    "Present_Price",
    "Driven_kms",
    "Owner"
]

categorical_features = [
    "Car_Name",
    "Fuel_Type",
    "Selling_type",
    "Transmission"
]


# 10. DATA PREPROCESSING

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median"))
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features)
    ]
)


# 11. TRAIN TEST SPLIT

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

print("\n========== TRAIN TEST SPLIT ==========")

print("Training records:", len(X_train))
print("Testing records:", len(X_test))


# 12. TRAIN MACHINE LEARNING MODELS

models = {
    "Linear Regression": LinearRegression(),

    "Decision Tree": DecisionTreeRegressor(
        random_state=42
    ),

    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        random_state=42
    )
}

results = []
trained_models = {}

for name, model in models.items():

    pipeline = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ]
    )

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_test)

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    mse = mean_squared_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(mse)

    r2 = r2_score(
        y_test,
        predictions
    )

    results.append({
        "Model": name,
        "MAE": mae,
        "MSE": mse,
        "RMSE": rmse,
        "R2 Score": r2
    })

    trained_models[name] = pipeline

    print("\n", name)

    print("MAE:", round(mae, 3))
    print("MSE:", round(mse, 3))
    print("RMSE:", round(rmse, 3))
    print("R2 Score:", round(r2, 3))


# 13. MODEL COMPARISON

results_df = pd.DataFrame(results)

print("\n========== MODEL COMPARISON ==========")

print(results_df.to_string(index=False))

best_model_name = results_df.sort_values(
    by="R2 Score",
    ascending=False
).iloc[0]["Model"]

best_model = trained_models[best_model_name]

print("\nBest Model:", best_model_name)


# 14. MODEL PERFORMANCE GRAPH

plt.figure(figsize=(9, 5))

sns.barplot(
    data=results_df,
    x="Model",
    y="R2 Score"
)

plt.title("Comparison of Machine Learning Models")
plt.xlabel("Model")
plt.ylabel("R2 Score")

plt.xticks(rotation=15)

plt.tight_layout()
plt.show()


# 15. ACTUAL VS PREDICTED PRICE

best_predictions = best_model.predict(X_test)

plt.figure(figsize=(8, 6))

plt.scatter(
    y_test,
    best_predictions,
    alpha=0.7
)

plt.xlabel("Actual Selling Price")
plt.ylabel("Predicted Selling Price")

plt.title("Actual vs Predicted Car Prices")

plt.tight_layout()
plt.show()


# 16. FEATURE IMPORTANCE (RANDOM FOREST)

rf_model = trained_models["Random Forest"]

feature_names = (
    rf_model.named_steps["preprocessor"]
    .get_feature_names_out()
)

importance = (
    rf_model.named_steps["model"]
    .feature_importances_
)

importance_df = pd.DataFrame({
    "Feature": feature_names,
    "Importance": importance
})

importance_df = importance_df.sort_values(
    by="Importance",
    ascending=False
).head(10)

print("\n========== TOP 10 IMPORTANT FEATURES ==========")

print(importance_df.to_string(index=False))

plt.figure(figsize=(10, 6))

sns.barplot(
    data=importance_df,
    x="Importance",
    y="Feature"
)

plt.title("Top 10 Important Features")

plt.tight_layout()
plt.show()


# 17. SAMPLE CAR PRICE PREDICTION

sample_car = X_test.iloc[[0]]

predicted_price = best_model.predict(sample_car)[0]

actual_price = y_test.iloc[0]

print("\n========== SAMPLE PREDICTION ==========")

print("\nCar Details:")
print(sample_car.to_string(index=False))

print("\nActual Selling Price:",
      round(actual_price, 2))

print("Predicted Selling Price:",
      round(predicted_price, 2))


# 18. FINAL PROJECT SUMMARY

print("\n========== FINAL PROJECT SUMMARY ==========")

print("Total Records:", len(df))

print("Number of Features:", X.shape[1])

print("Best Performing Model:", best_model_name)

best_r2 = results_df.loc[
    results_df["Model"] == best_model_name,
    "R2 Score"
].iloc[0]

print("Best Model R2 Score:", round(best_r2, 3))

print("\nProject Completed Successfully!")


# 19. SAVE CLEANED DATASET

cleaned_file = os.path.join(
    folder,
    "Cleaned_Car_Data.csv"
)

df.to_csv(cleaned_file, index=False)

print("\nCleaned dataset saved successfully!")

print("\n========== END OF PROJECT ==========")
