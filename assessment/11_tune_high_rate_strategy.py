from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import OneHotEncoder

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score,
    roc_auc_score
)

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")

OUTPUT_DIR = Path(
    "outputs/high_rate_tuning"
)

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
print("STEP 10 - HIGH RATE STRATEGY TUNING")
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
# Same successful baseline feature set
# ============================================================

def create_features(data):

    data = data.copy()

    # --------------------------------------------------------
    # DATE
    # --------------------------------------------------------

    data["year"] = data["date"].dt.year
    data["month"] = data["date"].dt.month
    data["day"] = data["date"].dt.day

    data["day_of_week"] = (
        data["date"].dt.dayofweek
    )

    data["week_of_year"] = (
        data["date"]
        .dt.isocalendar()
        .week
        .astype(int)
    )

    data["day_of_year"] = (
        data["date"].dt.dayofyear
    )

    data["is_weekend"] = (
        data["day_of_week"] >= 5
    ).astype(int)

    # --------------------------------------------------------
    # CYCLIC
    # --------------------------------------------------------

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

    safe_distance = data["distance"].replace(
        0,
        np.nan
    )

    data["weight_per_mile"] = (
        data["weight"]
        /
        safe_distance
    )

    data["distance_weight_interaction"] = (
        data["distance"]
        *
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
        -
        data["pickup_lat"]
    ).abs()

    data["lon_difference"] = (
        data["delivery_lon"]
        -
        data["pickup_lon"]
    ).abs()

    data["geo_distance"] = np.sqrt(
        (
            data["delivery_lat"]
            -
            data["pickup_lat"]
        ) ** 2
        +
        (
            data["delivery_lon"]
            -
            data["pickup_lon"]
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


print(
    "\nTraining rows:",
    len(train_df)
)

print(
    "Validation rows:",
    len(valid_df)
)


# ============================================================
# HIGH RATE LABEL
# ============================================================

train_df["is_high_rate"] = (
    train_df[TARGET]
    >= HIGH_RATE_THRESHOLD
).astype(int)

valid_df["is_high_rate"] = (
    valid_df[TARGET]
    >= HIGH_RATE_THRESHOLD
).astype(int)


print(
    "\nTraining high-rate:",
    train_df["is_high_rate"].sum()
)

print(
    "Validation high-rate:",
    valid_df["is_high_rate"].sum()
)


# ============================================================
# FEATURES
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
# CLASSIFIER
# ============================================================

print("\n" + "=" * 70)
print("TRAINING HIGH-RATE CLASSIFIER")
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

classifier.fit(
    X_train,
    y_high_train
)

high_probability = classifier.predict_proba(
    X_valid
)[:, 1]


auc = roc_auc_score(
    y_high_valid,
    high_probability
)

print(
    f"\nROC-AUC: {auc:.4f}"
)


# ============================================================
# BASE XGBOOST
# ============================================================

print("\n" + "=" * 70)
print("TRAINING BASE XGBOOST")
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
# BASELINE METRICS
# ============================================================

base_mae = mean_absolute_error(
    y_valid,
    base_predictions
)

base_rmse = np.sqrt(
    mean_squared_error(
        y_valid,
        base_predictions
    )
)

base_r2 = r2_score(
    y_valid,
    base_predictions
)


print("\n" + "=" * 70)
print("BASELINE")
print("=" * 70)

print(
    f"MAE  : ${base_mae:,.2f}"
)

print(
    f"RMSE : ${base_rmse:,.2f}"
)

print(
    f"R²   : {base_r2:.4f}"
)


# ============================================================
# TUNING GRID
# ============================================================

THRESHOLDS = [
    0.30,
    0.40,
    0.50,
    0.60,
    0.70,
    0.80,
    0.90
]

BOOSTS = [
    0.05,
    0.10,
    0.15,
    0.20,
    0.25,
    0.30
]


# ============================================================
# TEST STRATEGIES
# ============================================================

results = []

print("\n" + "=" * 70)
print("TESTING HIGH-RATE STRATEGIES")
print("=" * 70)

for threshold in THRESHOLDS:

    for boost in BOOSTS:

        predictions = base_predictions.copy()

        mask = (
            high_probability
            >= threshold
        )

        predictions[mask] = (
            predictions[mask]
            *
            (1 + boost)
        )

        mae = mean_absolute_error(
            y_valid,
            predictions
        )

        rmse = np.sqrt(
            mean_squared_error(
                y_valid,
                predictions
            )
        )

        r2 = r2_score(
            y_valid,
            predictions
        )

        high_mask = (
            y_valid
            >= HIGH_RATE_THRESHOLD
        )

        if high_mask.sum() > 0:

            high_mae = mean_absolute_error(
                y_valid[high_mask],
                predictions[high_mask]
            )

        else:

            high_mae = np.nan

        results.append({
            "threshold": threshold,
            "boost": boost,
            "MAE": mae,
            "RMSE": rmse,
            "R2": r2,
            "high_rate_MAE": high_mae,
            "triggered": int(mask.sum())
        })


# ============================================================
# RESULTS
# ============================================================

results_df = pd.DataFrame(
    results
)

results_df = results_df.sort_values(
    "MAE"
)


print("\n" + "=" * 70)
print("TOP STRATEGIES BY MAE")
print("=" * 70)

print(
    results_df
    .head(15)
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("TOP STRATEGIES BY RMSE")
print("=" * 70)

print(
    results_df
    .sort_values("RMSE")
    .head(15)
    .to_string(index=False)
)


print("\n" + "=" * 70)
print("TOP STRATEGIES FOR HIGH-RATE MAE")
print("=" * 70)

print(
    results_df
    .sort_values("high_rate_MAE")
    .head(15)
    .to_string(index=False)
)


# ============================================================
# BEST OVERALL
# ============================================================

best_mae = (
    results_df
    .sort_values("MAE")
    .iloc[0]
)

best_rmse = (
    results_df
    .sort_values("RMSE")
    .iloc[0]
)

best_high = (
    results_df
    .sort_values("high_rate_MAE")
    .iloc[0]
)


print("\n" + "=" * 70)
print("BEST STRATEGIES")
print("=" * 70)

print("\nBest overall MAE:")

print(
    best_mae.to_string()
)

print("\nBest overall RMSE:")

print(
    best_rmse.to_string()
)

print("\nBest high-rate MAE:")

print(
    best_high.to_string()
)


# ============================================================
# SAVE
# ============================================================

results_df.to_csv(
    OUTPUT_DIR /
    "high_rate_strategy_results.csv",
    index=False
)


print("\n" + "=" * 70)
print("STEP 10 COMPLETE")
print("=" * 70)

print(
    "\nResults saved to:"
)

print(
    OUTPUT_DIR /
    "high_rate_strategy_results.csv"
)