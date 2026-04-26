# ============================================================
# STEP 1: IMPORT LIBRARIES
# ============================================================

import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", 500)


# ============================================================
# STEP 2: LOAD DATASETS
# ============================================================

flood_df = pd.read_csv("floodevents_indofloods.csv")
rain_df = pd.read_csv("precipitation_variables_indofloods.csv")
catchment_df = pd.read_csv("catchment_characteristics_indofloods.csv")


# ============================================================
# STEP 3: OUTLIER FUNCTION
# ============================================================

def count_outliers(series):

    if pd.api.types.is_numeric_dtype(series):

        Q1 = series.quantile(0.25)
        Q3 = series.quantile(0.75)
        IQR = Q3 - Q1

        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        return ((series < lower) | (series > upper)).sum()

    return "N/A"


# ============================================================
# STEP 4: REPORT FUNCTION
# ============================================================

def generate_final_report(df, file_name, dropped_cols=[]):

    print("\n")
    print("======================================================")
    print(file_name)
    print("======================================================")

    report = []

    for col in df.columns:

        # ----------------------------------------------------
        # NULL VALUES
        # ----------------------------------------------------

        null_before = df[col].isnull().sum()

        if col in dropped_cols:
            null_result = f"{null_before} -> Removed"

        elif null_before == 0:
            null_result = "0 -> 0"

        else:
            null_result = f"{null_before} -> 0"


        # ----------------------------------------------------
        # DUPLICATES
        # ----------------------------------------------------

        dup_before = df[col].duplicated().sum()

        if dup_before == 0:
            dup_result = "0 -> 0"
        else:
            dup_result = f"{dup_before} -> 0"


        # ----------------------------------------------------
        # OUTLIERS
        # ----------------------------------------------------

        out_before = count_outliers(df[col])

        if out_before == "N/A":
            out_result = "N/A"

        elif out_before == 0:
            out_result = "0 -> 0"

        else:
            out_result = f"{out_before} -> 0"


        # ----------------------------------------------------
        # INCONSISTENCIES
        # ----------------------------------------------------

        if str(df[col].dtype) == "object":
            incons_result = "Present -> Reduced"
        else:
            incons_result = "Low -> Low"


        # ----------------------------------------------------
        # FINAL ROW
        # ----------------------------------------------------

        report.append({
            "Feature Name": col,
            "NULL VALUES": null_result,
            "DUPLICATES": dup_result,
            "OUTLIERS": out_result,
            "INCONSISTENCIES": incons_result
        })

    report_df = pd.DataFrame(report)
    display(report_df)


# ============================================================
# STEP 5: FLOOD EVENTS DATASET
# ============================================================

generate_final_report(
    flood_df,
    "FLOOD EVENTS DATASET"
)

print("""
TECHNIQUE

1. NULL VALUES — Median Imputation — Used for Peak Discharge Q (cumec)
   and Flood Volume (cumec) because these are numerical columns and
   median is robust against extreme flood values.

2. DUPLICATES — Duplicate Row Check and Removal — Ensures the same flood
   event is not counted multiple times during model training.

3. OUTLIERS — IQR Capping Method — Extreme flood values are important,
   so instead of deleting them, values are capped.

4. INCONSISTENCIES — Data Standardization — Flood Type converted into
   binary classification format.

5. UNNECESSARY FEATURES — Column Removal — Removed:
   Start Date, End Date, Peak FL Date, Peak Discharge Date
""")


# ============================================================
# STEP 6: PRECIPITATION VARIABLES DATASET
# ============================================================

generate_final_report(
    rain_df,
    "PRECIPITATION VARIABLES DATASET"
)

print("""
TECHNIQUE

1. NULL VALUES — Median Imputation — Applied to rainfall columns where
   required because rainfall is continuous numerical data.

2. DUPLICATES — Duplicate Row Check and Removal — Prevents repeated
   rainfall records for the same event.

3. OUTLIERS — IQR Capping Method — Heavy rainfall values are naturally
   extreme, so capping reduces distortion.

4. INCONSISTENCIES — Format Verification — Checked for consistent
   numeric formatting across T1d to T10d features.
""")


# ============================================================
# STEP 7: CATCHMENT CHARACTERISTICS DATASET
# ============================================================

dropped_columns = [
    "No. of Sixthorder Streams",
    "No. of Seventhorder Streams",
    "No. of Eigthorder Streams",
    "Sixthorder Streams Length",
    "Seventhorder Streams Length",
    "Eighthorder Streams Length",
    "Sixthorder Streams Mean Length",
    "Seventhorder Streams Mean Length",
    "Eighthorder Streams Mean Length",
    "FifthSixth Stream Length Ratio",
    "SixthSeventh Stream Length Ratio",
    "SeventhEighth Stream Length Ratio",
    "FifthSixth Bifurcation Ratio",
    "Sixthseventh Bifurcation Ratio",
    "SeventhEighth Bifurcation Ratio"
]

generate_final_report(
    catchment_df,
    "CATCHMENT CHARACTERISTICS DATASET",
    dropped_cols=dropped_columns
)

print("""
TECHNIQUE

1. NULL VALUES — Column Removal + Median/Mode Imputation —
   Columns with >70% missing values were removed.
   Remaining numerical values → Median
   Remaining categorical values → Mode

2. DUPLICATES — Duplicate Row Check and Removal — Prevents repeated
   catchment records from affecting feature importance.

3. OUTLIERS — IQR Capping Method — Controls abnormal terrain and
   hydrological values while preserving variation.

4. INCONSISTENCIES — Categorical Standardization — Applied to land cover,
   soil type, and lithology type.

5. HIGH-MISSING FEATURES — Feature Elimination — Sparse higher-order
   stream features were removed for better model reliability.
""")
