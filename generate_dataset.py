"""
Air Quality Dataset Generator
Creates a realistic atmospheric air quality dataset with real-world pollutant
correlations, meteorological factors, missing values, and duplicates.
"""

import numpy as np
import pandas as pd


def generate_air_quality_data(n_samples=1800, random_state=42):
    np.random.seed(random_state)

    # Categorical areas and seasons
    area_types = ["Industrial", "Commercial", "Residential", "Rural"]
    seasons = ["Winter", "Summer", "Monsoon", "Autumn"]

    area_choices = np.random.choice(
        area_types, size=n_samples, p=[0.25, 0.30, 0.30, 0.15]
    )
    season_choices = np.random.choice(
        seasons, size=n_samples, p=[0.25, 0.25, 0.25, 0.25]
    )

    # Area-specific pollutant multipliers
    area_multipliers = {
        "Industrial": 1.55,
        "Commercial": 1.25,
        "Residential": 0.85,
        "Rural": 0.50,
    }

    # Season-specific temperature and humidity baselines
    temp_list = []
    humidity_list = []
    for s in season_choices:
        if s == "Summer":
            temp_list.append(np.random.normal(36, 4))
            humidity_list.append(np.random.normal(45, 10))
        elif s == "Monsoon":
            temp_list.append(np.random.normal(29, 3))
            humidity_list.append(np.random.normal(82, 7))
        elif s == "Winter":
            temp_list.append(np.random.normal(16, 4))
            humidity_list.append(np.random.normal(60, 10))
        else:  # Autumn
            temp_list.append(np.random.normal(25, 3))
            humidity_list.append(np.random.normal(55, 8))

    temp = np.clip(np.array(temp_list), 5, 48)
    humidity = np.clip(np.array(humidity_list), 15, 98)

    # Pollutants generation
    pm25_list = []
    pm10_list = []
    no2_list = []
    so2_list = []
    co_list = []
    o3_list = []

    for i in range(n_samples):
        mult = area_multipliers[area_choices[i]]
        s = season_choices[i]

        # Inversion effect in winter increases particulate matter
        winter_boost = 1.35 if s == "Winter" else (0.75 if s == "Monsoon" else 1.0)

        # Base particulate matter
        base_pm25 = np.random.gamma(shape=3.5, scale=12.0) * mult * winter_boost
        base_pm10 = (
            base_pm25 * np.random.uniform(1.4, 2.1) + np.random.gamma(2, 8) * mult
        )

        # Gaseous pollutants
        base_no2 = np.random.gamma(shape=2.5, scale=10.0) * mult
        is_ind = area_choices[i] == "Industrial"
        so2_factor = (mult * 1.2) if is_ind else (mult * 0.7)
        base_so2 = np.random.gamma(shape=2.0, scale=8.0) * so2_factor
        base_co = np.random.gamma(shape=2.2, scale=0.45) * mult

        # Ground level ozone is formed in high heat and sunlight
        sunlight_boost = 1.4 if s == "Summer" else 0.8
        base_o3 = np.random.gamma(shape=3.0, scale=12.0) * sunlight_boost

        pm25_list.append(base_pm25)
        pm10_list.append(base_pm10)
        no2_list.append(base_no2)
        so2_list.append(base_so2)
        co_list.append(base_co)
        o3_list.append(base_o3)

    pm25 = np.clip(np.round(np.array(pm25_list), 1), 5.0, 350.0)
    pm10 = np.clip(np.round(np.array(pm10_list), 1), 10.0, 500.0)
    no2 = np.clip(np.round(np.array(no2_list), 1), 2.0, 180.0)
    so2 = np.clip(np.round(np.array(so2_list), 1), 1.0, 150.0)
    co = np.clip(np.round(np.array(co_list), 2), 0.1, 12.0)
    o3 = np.clip(np.round(np.array(o3_list), 1), 5.0, 220.0)
    temp = np.round(temp, 1)
    humidity = np.round(humidity, 1)

    # Calculate comprehensive AQI composite index
    aqi_raw = (
        0.35 * (pm25 / 35.0 * 100)
        + 0.25 * (pm10 / 75.0 * 100)
        + 0.15 * (no2 / 40.0 * 100)
        + 0.10 * (so2 / 40.0 * 100)
        + 0.10 * (co / 2.0 * 100)
        + 0.05 * (o3 / 100.0 * 100)
        + np.random.normal(0, 8, size=n_samples)
    )
    aqi_val = np.clip(np.round(aqi_raw, 1), 20.0, 480.0)

    # Binary Classification target:
    # 0 = Good / Moderate (AQI <= 100, Safe)
    # 1 = Poor / Hazardous (AQI > 100, Unhealthy / High Risk)
    status = (aqi_val > 100.0).astype(int)

    df = pd.DataFrame(
        {
            "PM2.5": pm25,
            "PM10": pm10,
            "NO2": no2,
            "SO2": so2,
            "CO": co,
            "O3": o3,
            "Temperature": temp,
            "Humidity": humidity,
            "Area_Type": area_choices,
            "Season": season_choices,
            "AQI_Value": aqi_val,
            "Air_Quality_Status": status,
        }
    )

    # Real-world data imperfections for Step 3 Preprocessing verification:
    # 1. Duplicate rows (~12 rows)
    dup_indices = np.random.choice(df.index, size=12, replace=False)
    duplicates = df.loc[dup_indices].copy()
    df = pd.concat([df, duplicates], ignore_index=True)

    # 2. Missing values (~15 missing entries in features)
    pollutant_cols = [
        "PM2.5",
        "PM10",
        "NO2",
        "SO2",
        "CO",
        "O3",
        "Temperature",
        "Humidity",
    ]
    for col in pollutant_cols:
        count = np.random.randint(1, 4)
        nan_indices = np.random.choice(df.index, size=count, replace=False)
        df.loc[nan_indices, col] = np.nan

    return df


if __name__ == "__main__":
    df = generate_air_quality_data(n_samples=1800, random_state=42)
    output_path = "air_quality_data.csv"
    df.to_csv(output_path, index=False)
    print(f"Generated realistic dataset saved to {output_path}")
    print(f"Total rows: {len(df)}, Columns: {list(df.columns)}")
    print("Class distribution:")
    print(df["Air_Quality_Status"].value_counts())
