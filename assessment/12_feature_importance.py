from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from sklearn.inspection import permutation_importance

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    r2_score
)

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")

OUTPUT_DIR = Path(
    "outputs/feature_importance"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TRAIN_FILE = DATA_DIR / "train_test.csv"

TARGET = "posted_rate"

SPLIT_DATE = "2025-09-01"


# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("STEP 11 - FEATURE IMPORTANCE")
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
# ORIGINAL SUCCESSFUL FEATURE SET
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
    # CYCLIC DATE FEATURES
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

train_df = df[
    df["date"] < SPLIT_DATE
].copy()

valid_df = df[
    df["date"] >= SPLIT_DATE
].copy()


# ============================================================
# X / Y
# ============================================================

DROP_COLUMNS = [
    TARGET,
    "load_id",
    "date"
]

X_train = train_df.drop(
    columns=DROP_COLUMNS
)

X_valid = valid_df.drop(
    columns=DROP_COLUMNS
)

y_train = train_df[TARGET]
y_valid = valid_df[TARGET]


# ============================================================
# FEATURE TYPES
# ============================================================

categorical_features = X_train.select_dtypes(
    include=["object", "category"]
).columns.tolist()

numeric_features = X_train.select_dtypes(
    include=["number", "bool"]
).columns.tolist()


print("\nNumerical features:")
print(len(numeric_features))

print("\nCategorical features:")
print(len(categorical_features))


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
# MODEL
# ============================================================

model = Pipeline(
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


# ============================================================
# TRAIN
# ============================================================

print("\n" + "=" * 70)
print("TRAINING MODEL")
print("=" * 70)

model.fit(
    X_train,
    np.log1p(y_train)
)


# ============================================================
# PREDICTION
# ============================================================

predictions = np.expm1(
    model.predict(X_valid)
)

predictions = np.maximum(
    predictions,
    0
)


# ============================================================
# METRICS
# ============================================================

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


print("\n" + "=" * 70)
print("MODEL PERFORMANCE")
print("=" * 70)

print(
    f"MAE  : ${mae:,.2f}"
)

print(
    f"RMSE : ${rmse:,.2f}"
)

print(
    f"R²   : {r2:.4f}"
)


# ============================================================
# XGBOOST NATIVE IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("XGBOOST FEATURE IMPORTANCE")
print("=" * 70)

fitted_preprocessor = (
    model.named_steps["preprocessor"]
)

fitted_model = (
    model.named_steps["model"]
)

feature_names = (
    fitted_preprocessor
    .get_feature_names_out()
)

importance_values = (
    fitted_model.feature_importances_
)

importance_df = pd.DataFrame({
    "feature": feature_names,
    "importance": importance_values
})

importance_df = importance_df.sort_values(
    "importance",
    ascending=False
)


print(
    importance_df
    .head(30)
    .to_string(index=False)
)


# ============================================================
# AGGREGATED IMPORTANCE
# ============================================================

importance_df[
    "original_feature"
] = (
    importance_df["feature"]
    .str.replace(
        "numeric__",
        "",
        regex=False
    )
    .str.replace(
        "categorical__",
        "",
        regex=False
    )
)


# For one-hot encoded categories,
# group by original feature prefix.

def get_group(feature):

    for original in (
        categorical_features
        +
        numeric_features
    ):

        if feature.startswith(
            original
        ):

            return original

    return feature


importance_df[
    "feature_group"
] = importance_df[
    "original_feature"
].apply(get_group)


grouped_importance = (
    importance_df
    .groupby("feature_group")
    ["importance"]
    .sum()
    .sort_values(
        ascending=False
    )
    .reset_index()
)


print("\n" + "=" * 70)
print("AGGREGATED FEATURE IMPORTANCE")
print("=" * 70)

print(
    grouped_importance
    .head(30)
    .to_string(index=False)
)


# ============================================================
# PERMUTATION IMPORTANCE
# ============================================================

print("\n" + "=" * 70)
print("PERMUTATION IMPORTANCE")
print("=" * 70)

# Use a sample for speed

sample_size = min(
    2500,
    len(X_valid)
)

sample_indices = np.random.RandomState(
    42
).choice(
    len(X_valid),
    size=sample_size,
    replace=False
)

X_sample = X_valid.iloc[
    sample_indices
]

y_sample = y_valid.iloc[
    sample_indices
]


print(
    "\nPermutation sample size:",
    sample_size
)

permutation = permutation_importance(
    model,
    X_sample,
    y_sample,
    scoring="neg_mean_absolute_error",
    n_repeats=5,
    random_state=42,
    n_jobs=-1
)

permutation_df = pd.DataFrame({
    "feature": X_valid.columns,
    "importance_mean": (
        permutation.importances_mean
    ),
    "importance_std": (
        permutation.importances_std
    )
})

permutation_df = permutation_df.sort_values(
    "importance_mean",
    ascending=False
)


print(
    permutation_df
    .head(30)
    .to_string(index=False)
)


# ============================================================
# SAVE
# ============================================================

importance_df.to_csv(
    OUTPUT_DIR /
    "xgboost_feature_importance.csv",
    index=False
)

grouped_importance.to_csv(
    OUTPUT_DIR /
    "aggregated_feature_importance.csv",
    index=False
)

permutation_df.to_csv(
    OUTPUT_DIR /
    "permutation_importance.csv",
    index=False
)


print("\n" + "=" * 70)
print("STEP 11 COMPLETE")
print("=" * 70)

print(
    "\nFiles saved to:"
)

print(
    OUTPUT_DIR.resolve()
)