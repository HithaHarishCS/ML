# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 200)


# ============================================================
# STEP 2: UPLOAD FILES
# ============================================================

from google.colab import files
uploaded = files.upload()


# ============================================================
# STEP 3: LOAD DATASETS
# ============================================================

flood_df = pd.read_csv("floodevents_indofloods.csv")
rain_df = pd.read_csv("precipitation_variables_indofloods.csv")
catchment_df = pd.read_csv("catchment_characteristics_indofloods.csv")


# ============================================================
# STEP 4: DISPLAY FIRST 3 RECORDS + SHAPE
# ============================================================

def show_basic_info(df, name):
    print("\n")
    print("===================================================")
    print(f"{name}")
    print("===================================================")

    print("\nFirst 3 Records:\n")
    display(df.head(3))

    print("\nShape of Dataset:")
    print(df.shape)


show_basic_info(flood_df, "Flood Events Dataset")
show_basic_info(rain_df, "Precipitation Variables Dataset")
show_basic_info(catchment_df, "Catchment Characteristics Dataset")


# ============================================================
# STEP 5: DATA QUALITY REPORT TABLE
# ============================================================

def data_quality_report(df, dataset_name):
    print("\n")
    print("===================================================")
    print(f"DATA QUALITY REPORT : {dataset_name}")
    print("===================================================")

    report = pd.DataFrame()

    # Feature Name
    report["Feature"] = df.columns

    # Null Values
    report["Null Values"] = df.isnull().sum().values

    # Repeated Values
    report["Repeated Values"] = [
        df[col].duplicated().sum() for col in df.columns
    ]

    # Missing Data
    report["Missing Data"] = np.where(
        report["Null Values"] > 0,
        "Yes",
        "No"
    )

    # Outlier Detection using IQR
    outlier_list = []

    for col in df.columns:
        if pd.api.types.is_numeric_dtype(df[col]):

            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1

            lower = Q1 - 1.5 * IQR
            upper = Q3 + 1.5 * IQR

            outliers = df[
                (df[col] < lower) | (df[col] > upper)
            ].shape[0]

            outlier_list.append(outliers)

        else:
            outlier_list.append("N/A")

    report["Outliers"] = outlier_list

    # Noise Detection
    report["Noise"] = np.where(
        report["Repeated Values"] > (len(df) * 0.8),
        "Possible",
        "Low"
    )

    # Inconsistencies
    report["Inconsistencies"] = np.where(
        df.dtypes.astype(str).values == "object",
        "Check formatting/manual review",
        "Low"
    )

    # Data Format
    report["Data Format"] = df.dtypes.astype(str).values

    display(report)


data_quality_report(flood_df, "Flood Events Dataset")
data_quality_report(rain_df, "Precipitation Variables Dataset")
data_quality_report(catchment_df, "Catchment Characteristics Dataset")
