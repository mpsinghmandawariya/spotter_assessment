from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    roc_auc_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")
OUTPUT_DIR = Path("outputs/two_stage")

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_FILE = DATA_DIR / "train_test.csv"

TARGET = "posted_rate"

SPLIT_DATE = "2025-09-01"

HIGH_RATE_THRESHOLD = 7500


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("STEP 9 - TWO-STAGE HIGH-RATE MODEL")
print("=" * 70)

df = pd.read_csv(
    TRAIN_FILE
)

df["date"] = pd.to_datetime(
    df["date"],
    errors="coerce"
)


# ============================================================
# FEATURE ENGINEERING
# Use the ORIGINAL successful feature set
# ============================================================

def create_features(data):

    data = data.copy()

    # --------------------------------------------------------
    # DATE
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

    # Cyclic features

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
    # GEOGRAPHY
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
# TIME SPLIT
# ============================================================

split_date = pd.Timestamp(
    SPLIT_DATE
)

train_df = df[
    df["date"] < split_date
].copy()

valid_df = df[
    df["date"] >= split_date
].copy()


print("\nTraining rows:", len(train_df))
print("Validation rows:", len(valid_df))


# ============================================================
# CREATE HIGH-RATE LABEL
# ============================================================

train_df["is_high_rate"] = (
    train_df[TARGET]
    >= HIGH_RATE_THRESHOLD
).astype(int)

valid_df["is_high_rate"] = (
    valid_df[TARGET]
    >= HIGH_RATE_THRESHOLD
).astype(int)


print("\n" + "=" * 70)
print("HIGH-RATE DISTRIBUTION")
print("=" * 70)

print(
    "Training high-rate loads:",
    train_df["is_high_rate"].sum()
)

print(
    "Validation high-rate loads:",
    valid_df["is_high_rate"].sum()
)

print(
    "Training high-rate %:",
    train_df["is_high_rate"].mean() * 100
)

print(
    "Validation high-rate %:",
    valid_df["is_high_rate"].mean() * 100
)


# ============================================================
# PREPARE FEATURES
# ============================================================

DROP_COLUMNS = [
    TARGET,
    "load_id",
    "date",
    "is_high_rate"
]

X_train = train_df.drop(
    columns=DROP_COLUMNS
)

X_valid = valid_df.drop(
    columns=DROP_COLUMNS
)

y_train = train_df[TARGET]
y_valid = valid_df[TARGET]

y_high_train = train_df[
    "is_high_rate"
]

y_high_valid = valid_df[
    "is_high_rate"
]


# ============================================================
# FEATURE TYPES
# ============================================================

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_features = X_train.select_dtypes(
    include=["number", "bool"]
).columns.tolist()


# ============================================================
# PREPROCESSOR
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
# STAGE 1 - HIGH RATE CLASSIFIER
# ============================================================

print("\n" + "=" * 70)
print("STAGE 1 - HIGH RATE CLASSIFIER")
print("=" * 70)

classifier = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            RandomForestClassifier(
                n_estimators=300,
                max_depth=12,
                min_samples_leaf=3,
                class_weight="balanced",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("\nTraining classifier...")

classifier.fit(
    X_train,
    y_high_train
)

high_probability = classifier.predict_proba(
    X_valid
)[:, 1]

high_prediction = (
    high_probability >= 0.5
).astype(int)


# ============================================================
# CLASSIFIER METRICS
# ============================================================

print("\nClassification Report:")

print(
    classification_report(
        y_high_valid,
        high_prediction,
        zero_division=0
    )
)

try:

    auc = roc_auc_score(
        y_high_valid,
        high_probability
    )

    print(
        f"ROC-AUC: {auc:.4f}"
    )

except ValueError:

    print(
        "ROC-AUC could not be calculated."
    )


# ============================================================
# STAGE 2 - REGRESSION MODEL
# ============================================================

print("\n" + "=" * 70)
print("STAGE 2 - XGBOOST REGRESSION")
print("=" * 70)

regressor = Pipeline(
    steps=[
        (
            "preprocessor",
            preprocessor
        ),
        (
            "model",
            XGBRegressor(
                n_estimators=700,
                learning_rate=0.05,
                max_depth=8,
                min_child_weight=5,
                subsample=0.85,
                colsample_bytree=0.85,
                reg_alpha=0.1,
                reg_lambda=1.0,
                objective="reg:squarederror",
                random_state=42,
                n_jobs=-1
            )
        )
    ]
)

print("\nTraining regression model...")

regressor.fit(
    X_train,
    np.log1p(y_train)
)

log_predictions = regressor.predict(
    X_valid
)

base_predictions = np.expm1(
    log_predictions
)

base_predictions = np.maximum(
    base_predictions,
    0
)


# ============================================================
# TWO-STAGE PREDICTION
# ============================================================

# First version:
# Use classifier probability to boost predictions
# only when high-rate probability is strong.

boost_factor = np.clip(
    high_probability,
    0,
    1
)

# We don't want to blindly multiply normal predictions.
# Instead, create a high-rate adjustment.

two_stage_predictions = (
    base_predictions
    *
    (
        1
        +
        1.5 * boost_factor
    )
)


# ============================================================
# EVALUATE BASELINE
# ============================================================

baseline_mae = mean_absolute_error(
    y_valid,
    base_predictions
)

baseline_rmse = np.sqrt(
    mean_squared_error(
        y_valid,
        base_predictions
    )
)

baseline_r2 = r2_score(
    y_valid,
    base_predictions
)


# ============================================================
# EVALUATE TWO-STAGE
# ============================================================

two_stage_mae = mean_absolute_error(
    y_valid,
    two_stage_predictions
)

two_stage_rmse = np.sqrt(
    mean_squared_error(
        y_valid,
        two_stage_predictions
    )
)

two_stage_r2 = r2_score(
    y_valid,
    two_stage_predictions
)


# ============================================================
# RESULTS
# ============================================================

print("\n" + "=" * 70)
print("MODEL COMPARISON")
print("=" * 70)

results = pd.DataFrame([
    {
        "model": "XGBoost Log Baseline",
        "MAE": baseline_mae,
        "RMSE": baseline_rmse,
        "R2": baseline_r2
    },
    {
        "model": "Two Stage",
        "MAE": two_stage_mae,
        "RMSE": two_stage_rmse,
        "R2": two_stage_r2
    }
])

print(
    results.to_string(
        index=False
    )
)


# ============================================================
# HIGH-RATE PERFORMANCE
# ============================================================

high_mask = (
    y_valid >= HIGH_RATE_THRESHOLD
)

print("\n" + "=" * 70)
print("HIGH-RATE LOAD PERFORMANCE")
print("=" * 70)

if high_mask.sum() > 0:

    baseline_high_mae = mean_absolute_error(
        y_valid[high_mask],
        base_predictions[high_mask]
    )

    two_stage_high_mae = mean_absolute_error(
        y_valid[high_mask],
        two_stage_predictions[high_mask]
    )

    print(
        "High-rate observations:",
        high_mask.sum()
    )

    print(
        f"Baseline high-rate MAE: "
        f"${baseline_high_mae:,.2f}"
    )

    print(
        f"Two-stage high-rate MAE: "
        f"${two_stage_high_mae:,.2f}"
    )

else:

    print(
        "No high-rate observations in validation."
    )


# ============================================================
# TOP ERRORS
# ============================================================

error_df = valid_df[
    [
        "load_id",
        "date",
        "pickup",
        "delivery",
        "equipment",
        "distance",
        TARGET
    ]
].copy()

error_df["baseline_prediction"] = (
    base_predictions
)

error_df["two_stage_prediction"] = (
    two_stage_predictions
)

error_df["high_probability"] = (
    high_probability
)

error_df["baseline_error"] = (
    error_df["baseline_prediction"]
    - error_df[TARGET]
).abs()

error_df["two_stage_error"] = (
    error_df["two_stage_prediction"]
    - error_df[TARGET]
).abs()


print("\n" + "=" * 70)
print("TOP TWO-STAGE ERRORS")
print("=" * 70)

print(
    error_df
    .sort_values(
        "two_stage_error",
        ascending=False
    )
    .head(20)
    .to_string(
        index=False
    )
)


# ============================================================
# SAVE
# ============================================================

results.to_csv(
    OUTPUT_DIR /
    "two_stage_results.csv",
    index=False
)

error_df.to_csv(
    OUTPUT_DIR /
    "two_stage_predictions.csv",
    index=False
)


print("\n" + "=" * 70)
print("STEP 9 COMPLETE")
print("=" * 70)