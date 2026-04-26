# ============================================================
# IMPROVED RANDOM FOREST MODEL
# Removed Target Leakage
#
# Tasks:
# 1. Flood Occurrence Prediction (Yes / No)
# 2. Flood Severity Prediction (Low / Medium / High)
# 3. Flood Time Prediction (Regression)
#
# Split:
# Train = 70%
# Validation = 15%
# Test = 15%
#
# IMPORTANT:
# peak_water_level, peak_discharge, flood_volume,
# time_to_peak are NOT used directly for prediction
# because they create target leakage.
# ============================================================


# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from sklearn.ensemble import (
    RandomForestClassifier,
    RandomForestRegressor
)

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)


# ============================================================
# STEP 2: LOAD FINAL FEATURE ENGINEERED DATASET
# ============================================================

df = pd.read_csv("final_feature_engineered_dataset.csv")

print("Dataset Loaded Successfully")
print("Shape:", df.shape)


# ============================================================
# STEP 3: CREATE TARGET VARIABLES
# ============================================================

# ------------------------------------------------------------
# Target 1 : Flood Occurrence
# Based on flood_volume + peak_discharge threshold
# ------------------------------------------------------------

df["flood_occurrence"] = np.where(
    (
        (df["flood_volume"] > df["flood_volume"].median()) |
        (df["peak_discharge"] > df["peak_discharge"].median())
    ),
    1,
    0
)


# ------------------------------------------------------------
# Target 2 : Flood Severity
# Based on flood_volume
# ------------------------------------------------------------

conditions = [
    df["flood_volume"] <= df["flood_volume"].quantile(0.33),

    (df["flood_volume"] > df["flood_volume"].quantile(0.33)) &
    (df["flood_volume"] <= df["flood_volume"].quantile(0.66)),

    df["flood_volume"] > df["flood_volume"].quantile(0.66)
]

values = ["Low", "Medium", "High"]

df["flood_severity"] = np.select(
    conditions,
    values,
    default="Medium"
)

label_encoder = LabelEncoder()
df["flood_severity"] = label_encoder.fit_transform(
    df["flood_severity"]
)


# ------------------------------------------------------------
# Target 3 : Flood Time Prediction
# ------------------------------------------------------------

df["flood_time_prediction"] = df["time_to_peak"]


# ============================================================
# STEP 4: REMOVE LEAKAGE COLUMNS
# ============================================================

X = df.drop([
    # targets
    "flood_occurrence",
    "flood_severity",
    "flood_time_prediction",

    # direct leakage columns
    "peak_water_level",
    "peak_discharge",
    "flood_volume",
    "time_to_peak",

    # duplicate leakage columns
    "river_water_level",
    "discharge_rate",
    "streamflow"
], axis=1)

# categorical encoding
X = pd.get_dummies(X, drop_first=True)

print("\nFinal Feature Columns Used:")
print(X.columns.tolist())


# ============================================================
# STEP 5: TRAIN / VALIDATION / TEST SPLIT
# 70 : 15 : 15
# ============================================================

X_train, X_temp = train_test_split(
    X,
    test_size=0.30,
    random_state=42
)

X_val, X_test = train_test_split(
    X_temp,
    test_size=0.50,
    random_state=42
)

print("\nSplit Completed")
print("Train Shape :", X_train.shape)
print("Validation Shape :", X_val.shape)
print("Test Shape :", X_test.shape)


# ============================================================
# MODEL 1 : RANDOM FOREST CLASSIFIER
# FLOOD OCCURRENCE
# ============================================================

print("\n================================================")
print("RANDOM FOREST - FLOOD OCCURRENCE")
print("================================================")

y = df["flood_occurrence"]

y_train, y_temp = train_test_split(
    y,
    test_size=0.30,
    random_state=42
)

y_val, y_test = train_test_split(
    y_temp,
    test_size=0.50,
    random_state=42
)

rf_occurrence = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

rf_occurrence.fit(X_train, y_train)

pred = rf_occurrence.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n")
print(classification_report(y_test, pred))


# ============================================================
# MODEL 2 : RANDOM FOREST CLASSIFIER
# FLOOD SEVERITY
# ============================================================

print("\n================================================")
print("RANDOM FOREST - FLOOD SEVERITY")
print("================================================")

y = df["flood_severity"]

y_train, y_temp = train_test_split(
    y,
    test_size=0.30,
    random_state=42
)

y_val, y_test = train_test_split(
    y_temp,
    test_size=0.50,
    random_state=42
)

rf_severity = RandomForestClassifier(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

rf_severity.fit(X_train, y_train)

pred = rf_severity.predict(X_test)

print("Accuracy:", accuracy_score(y_test, pred))
print("\nClassification Report:\n")
print(classification_report(y_test, pred))


# ============================================================
# MODEL 3 : RANDOM FOREST REGRESSOR
# FLOOD TIME PREDICTION
# ============================================================

print("\n================================================")
print("RANDOM FOREST REGRESSOR - FLOOD TIME")
print("================================================")

y = df["flood_time_prediction"]

y_train, y_temp = train_test_split(
    y,
    test_size=0.30,
    random_state=42
)

y_val, y_test = train_test_split(
    y_temp,
    test_size=0.50,
    random_state=42
)

rf_regressor = RandomForestRegressor(
    n_estimators=200,
    max_depth=8,
    random_state=42
)

rf_regressor.fit(X_train, y_train)

pred = rf_regressor.predict(X_test)

print("MAE :", mean_absolute_error(y_test, pred))
print("MSE :", mean_squared_error(y_test, pred))
print("R2 Score :", r2_score(y_test, pred))
