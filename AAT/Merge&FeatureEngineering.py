import pandas as pd
import numpy as np

pd.set_option("display.max_columns", None)

flood_df = pd.read_csv("cleaned_flood_events.csv")
rain_df = pd.read_csv("cleaned_precipitation.csv")
catchment_df = pd.read_csv("cleaned_catchment.csv")

print("Datasets loaded successfully.\n")


# ============================================================
# STEP 3: DISPLAY HOW DATASETS ARE MERGED
# ============================================================

print("===================================================")
print("Merging Datsets")
print("===================================================\n")

print("1. Flood Events + Precipitation")
print("→ Common Key Used: EventID")

print("Flood EventID  : INDOFLOODS-gauge-1010-1")
print("Rain EventID   : INDOFLOODS-gauge-1010-1")


print("---------------------------------------------------")

print("\n2. Flood Events + Catchment")
print("→ Common Key Used: GaugeID")

print("\nBut Flood dataset does NOT have direct GaugeID.")
print("It is extracted from EventID.")

print("\nExample:")
print("EventID  : INDOFLOODS-gauge-1010-1")
print("GaugeID  : INDOFLOODS-gauge-1010")

flood_df["GaugeID"] = flood_df["EventID"].str.rsplit("-", n=1).str[0]

# ============================================================
# STEP 5: MERGE FLOOD + RAIN USING EventID
# ============================================================

merged_df = pd.merge(
    flood_df,
    rain_df,
    on="EventID",
    how="inner"
)

# ============================================================
# STEP 6: MERGE WITH CATCHMENT USING GaugeID
# ============================================================

merged_df = pd.merge(
    merged_df,
    catchment_df,
    on="GaugeID",
    how="left"
)

# ------------------------------------------------------------
# FLOOD EVENT FEATURES
# ------------------------------------------------------------

final_df = pd.DataFrame()

final_df["peak_water_level"] = merged_df["Peak Flood Level (m)"]

final_df["peak_discharge"] = merged_df["Peak Discharge Q (cumec)"]

final_df["event_duration"] = merged_df["Event Duration (days)"]

final_df["time_to_peak"] = merged_df["Time to Peak (days)"]

final_df["flood_volume"] = merged_df["Flood Volume (cumec)"]

# Derived Feature
final_df["flood_frequency"] = (
    merged_df.groupby("GaugeID")["GaugeID"]
    .transform("count")
)

# Derived Feature
final_df["flood_return_period"] = (
    len(merged_df) / final_df["flood_frequency"]
)


# ------------------------------------------------------------
# RIVER FLOW / HYDROLOGICAL VARIABLES
# ------------------------------------------------------------

final_df["river_water_level"] = merged_df["Peak Flood Level (m)"]

final_df["discharge_rate"] = merged_df["Peak Discharge Q (cumec)"]

# streamflow approximated using discharge
final_df["streamflow"] = merged_df["Peak Discharge Q (cumec)"]

# groundwater not available → approximated using antecedent rainfall later


# ------------------------------------------------------------
# PRECIPITATION VARIABLES
# ------------------------------------------------------------

final_df["event_rainfall"] = merged_df["T10d"]

final_df["antecedent_rainfall"] = (
    merged_df["T1d"] +
    merged_df["T2d"] +
    merged_df["T3d"]
)

final_df["rainfall_intensity"] = (
    merged_df["T10d"] / 10
)

final_df["seasonal_rainfall_pattern"] = (
    merged_df["T10d"] - merged_df["T1d"]
)

# groundwater proxy
final_df["groundwater_level"] = final_df["antecedent_rainfall"]


# ------------------------------------------------------------
# CATCHMENT FEATURES
# ------------------------------------------------------------

final_df["elevation"] = merged_df["Catchment Relief"]

final_df["slope"] = merged_df["Relief Ratio"]

final_df["drainage_density"] = merged_df["Drainage Density"]

final_df["land_cover"] = merged_df["Land cover"]

final_df["soil_type"] = merged_df["Soil type"]

final_df["lithology"] = merged_df["lithology type"]

final_df["urbanization_index"] = merged_df["Urban percentage"]

# Reservoir approximation
final_df["reservoir_presence"] = np.where(
    merged_df["Drainage Area"] >
    merged_df["Drainage Area"].median(),
    1,
    0
)


# ============================================================
# STEP 8: DISPLAY FINAL FEATURES
# ============================================================

print("\nFeature Engineered Dataset\n")

print("Shape:")
print(final_df.shape)

display(final_df.head(3))


# ============================================================
# STEP 9: SAVE FINAL DATASET
# ============================================================

final_df.to_csv(
    "final_feature_engineered_dataset.csv",
    index=False
)

print("\nSaved Successfully:")
print("final_feature_engineered_dataset.csv")
