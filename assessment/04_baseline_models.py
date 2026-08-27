from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs")

TRAIN_FILE = DATA_DIR / "train_test.csv"

TARGET = "posted_rate"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("STEP 4 - VALIDATION + BASELINE MODELS")
print("=" * 70)

df = pd.read_csv(TRAIN_FILE)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


# ============================================================
# BASIC FEATURE ENGINEERING
# ============================================================

def create_features(data):

    data = data.copy()

    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day
    data["day_of_week"] = data["date"].dt.dayofweek
    data["day_of_year"] = data["date"].dt.dayofyear

    data["week_of_year"] = (
        data["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    # Cyclic date features

    data["month_sin"] = np.sin(
        2 * np.pi * data["month"] / 12
    )

    data["month_cos"] = np.cos(
        2 * np.pi * data["month"] / 12
    )

    data["weekday_sin"] = np.sin(
        2 * np.pi * data["day_of_week"] / 7
    )

    data["weekday_cos"] = np.cos(
        2 * np.pi * data["day_of_week"] / 7
    )

    # --------------------------------------------------------
    # ROUTE
    # --------------------------------------------------------

    data["route"] = (
        data["pickup"].astype(str)
        + " → "
        + data["delivery"].astype(str)
    )

    # --------------------------------------------------------
    # DISTANCE
    # --------------------------------------------------------

    data["distance_log"] = np.log1p(
        data["distance"].clip(lower=0)
    )

    data["distance_squared"] = (
        data["distance"] ** 2
    )

    # --------------------------------------------------------
    # WEIGHT
    # --------------------------------------------------------

    data["weight_log"] = np.log1p(
        data["weight"].clip(lower=0)
    )

    data["weight_squared"] = (
        data["weight"] ** 2
    )

    # --------------------------------------------------------
    # WEIGHT / DISTANCE
    # --------------------------------------------------------

    safe_distance = data["distance"].replace(
        0,
        np.nan
    )

    data["weight_per_mile"] = (
        data["weight"] /
        safe_distance
    )

    data["distance_weight_interaction"] = (
        data["distance"] *
        data["weight"]
    )

    # --------------------------------------------------------
    # MARKET
    # --------------------------------------------------------

    data["market_index_squared"] = (
        data["market_index"] ** 2
    )

    data["market_index_log"] = np.log1p(
        data["market_index"].clip(lower=0)
    )

    # --------------------------------------------------------
    # QUOTE SIGNAL
    # --------------------------------------------------------

    data["quote_signal_squared"] = (
        data["quote_signal"] ** 2
    )

    data["quote_signal_abs"] = (
        data["quote_signal"].abs()
    )

    # --------------------------------------------------------
    # GEOGRAPHIC FEATURES
    # --------------------------------------------------------

    data["lat_difference"] = (
        data["delivery_lat"]
        - data["pickup_lat"]
    ).abs()

    data["lon_difference"] = (
        data["delivery_lon"]
        - data["pickup_lon"]
    ).abs()

    data["geo_distance"] = np.sqrt(
        (
            data["delivery_lat"]
            - data["pickup_lat"]
        ) ** 2
        +
        (
            data["delivery_lon"]
            - data["pickup_lon"]
        ) ** 2
    )

    return data


df = create_features(df)


# ============================================================
# REMOVE TARGET / ID / RAW DATE
# ============================================================

DROP_COLUMNS = [
    TARGET,
    "load_id",
    "date"
]

X = df.drop(
    columns=DROP_COLUMNS
)

y = df[TARGET]


# ============================================================
# IDENTIFY FEATURE TYPES
# ============================================================

categorical_features = X.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_features = X.select_dtypes(
    include=["number", "bool"]
).columns.tolist()


print("\nCategorical features:")
print(categorical_features)

print("\nNumerical feature count:")
print(len(numeric_features))


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="median"
            )
        )
    ]
)

categorical_pipeline = Pipeline(
    steps=[
        (
            "imputer",
            SimpleImputer(
                strategy="most_frequent"
            )
        ),
        (
            "encoder",
            OneHotEncoder(
                handle_unknown="ignore"
            )
        )
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        (
            "numeric",
            numeric_pipeline,
            numeric_features
        ),
        (
            "categorical",
            categorical_pipeline,
            categorical_features
        )
    ]
)


# ============================================================
# METRICS
# ============================================================

def evaluate_model(
    name,
    y_true,
    predictions
):

    mae = mean_absolute_error(
        y_true,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_true,
            predictions
        )
    )

    r2 = r2_score(
        y_true,
        predictions
    )

    print(
        f"\n{name}"
    )

    print(
        f"MAE  : ${mae:,.2f}"
    )

    print(
        f"RMSE : ${rmse:,.2f}"
    )

    print(
        f"R²   : {r2:.4f}"
    )

    return {
        "model": name,
        "MAE": mae,
        "RMSE": rmse,
        "R2": r2
    }


# ============================================================
# SPLIT 1 - RANDOM 80/20
# ============================================================

print("\n" + "=" * 70)
print("RANDOM 80/20 VALIDATION")
print("=" * 70)

from sklearn.model_selection import train_test_split

X_train_random, X_valid_random, y_train_random, y_valid_random = (
    train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42
    )
)


# ============================================================
# RIDGE - RANDOM SPLIT
# ============================================================

ridge_random = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            Ridge(alpha=10.0)
        )
    ]
)

print("\nTraining Ridge - Random Split...")

ridge_random.fit(
    X_train_random,
    y_train_random
)

ridge_pred = ridge_random.predict(
    X_valid_random
)

random_ridge_results = evaluate_model(
    "Ridge - Random Split",
    y_valid_random,
    ridge_pred
)


# ============================================================
# RANDOM FOREST - RANDOM SPLIT
# ============================================================

rf_random = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("\nTraining Random Forest - Random Split...")

rf_random.fit(
    X_train_random,
    y_train_random
)

rf_pred = rf_random.predict(
    X_valid_random
)

random_rf_results = evaluate_model(
    "Random Forest - Random Split",
    y_valid_random,
    rf_pred
)


# ============================================================
# SPLIT 2 - TIME BASED
# ============================================================

print("\n" + "=" * 70)
print("TIME-BASED VALIDATION")
print("=" * 70)

# Sort chronologically

df_sorted = df.sort_values(
    "date"
).reset_index(
    drop=True
)

split_date = pd.Timestamp(
    "2025-09-01"
)

train_time = df_sorted[
    df_sorted["date"] < split_date
]

valid_time = df_sorted[
    df_sorted["date"] >= split_date
]


X_train_time = train_time.drop(
    columns=DROP_COLUMNS
)

y_train_time = train_time[TARGET]

X_valid_time = valid_time.drop(
    columns=DROP_COLUMNS
)

y_valid_time = valid_time[TARGET]


print(
    f"\nTraining period:"
    f" {train_time['date'].min().date()}"
    f" → {train_time['date'].max().date()}"
)

print(
    f"Validation period:"
    f" {valid_time['date'].min().date()}"
    f" → {valid_time['date'].max().date()}"
)

print(
    f"\nTraining rows  : {len(train_time):,}"
)

print(
    f"Validation rows: {len(valid_time):,}"
)


# ============================================================
# RIDGE - TIME SPLIT
# ============================================================

ridge_time = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            Ridge(alpha=10.0)
        )
    ]
)

print("\nTraining Ridge - Time Split...")

ridge_time.fit(
    X_train_time,
    y_train_time
)

ridge_time_pred = ridge_time.predict(
    X_valid_time
)

time_ridge_results = evaluate_model(
    "Ridge - Time Split",
    y_valid_time,
    ridge_time_pred
)


# ============================================================
# RANDOM FOREST - TIME SPLIT
# ============================================================

rf_time = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestRegressor(
                n_estimators=200,
                max_depth=20,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("\nTraining Random Forest - Time Split...")

rf_time.fit(
    X_train_time,
    y_train_time
)

rf_time_pred = rf_time.predict(
    X_valid_time
)

time_rf_results = evaluate_model(
    "Random Forest - Time Split",
    y_valid_time,
    rf_time_pred
)


# ============================================================
# COMPARE RESULTS
# ============================================================

results = pd.DataFrame([
    random_ridge_results,
    random_rf_results,
    time_ridge_results,
    time_rf_results
])

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

print(
    results.sort_values(
        "RMSE"
    ).to_string(
        index=False
    )
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    OUTPUT_DIR / "baseline_results.csv",
    index=False
)

print(
    "\nResults saved to:"
    f" {OUTPUT_DIR / 'baseline_results.csv'}"
)

print("\nBASELINE EXPERIMENT COMPLETE.")