from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from xgboost import XGBRegressor


# ============================================================
# CONFIGURATION
# ============================================================

TRAIN_FILE = Path("data/train_test.csv")
VALIDATION_FILE = Path("data/validation.csv")
TEMPLATE_FILE = Path(
    "data/validation-predictions-template.csv"
)

OUTPUT_DIR = Path(
    "outputs/final_model"
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

TARGET = "posted_rate"


# ============================================================
# LOCKED MODEL CONFIGURATION
# ============================================================

PARAMS = {
    "n_estimators": 700,
    "learning_rate": 0.05,
    "max_depth": 6,
    "min_child_weight": 5,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "objective": "reg:squarederror",
    "random_state": 42,
    "n_jobs": -1
}


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 16 - FINAL MODEL TRAINING")
print("=" * 70)


# ============================================================
# LOAD DATA
# ============================================================

train = pd.read_csv(
    TRAIN_FILE
)

validation = pd.read_csv(
    VALIDATION_FILE
)

template = pd.read_csv(
    TEMPLATE_FILE
)


print("\nDATASET INFORMATION")
print("-" * 70)

print(
    "Training shape:",
    train.shape
)

print(
    "Prediction shape:",
    validation.shape
)

print(
    "Template shape:",
    template.shape
)


# ============================================================
# BASIC VALIDATION
# ============================================================

assert TARGET in train.columns, (
    f"{TARGET} missing from training data"
)

assert TARGET not in validation.columns, (
    f"{TARGET} unexpectedly present in validation data"
)

assert len(validation) == len(template), (
    "Validation and template row counts do not match"
)

assert validation["load_id"].equals(
    template["load_id"]
), (
    "Validation load_id order does not match template"
)


# ============================================================
# DATE CONVERSION
# ============================================================

train["date"] = pd.to_datetime(
    train["date"],
    errors="coerce"
)

validation["date"] = pd.to_datetime(
    validation["date"],
    errors="coerce"
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(data):

    data = data.copy()

    # --------------------------------------------------------
    # DATE FEATURES
    # --------------------------------------------------------

    data["year"] = (
        data["date"].dt.year
    )

    data["month"] = (
        data["date"].dt.month
    )

    data["day"] = (
        data["date"].dt.day
    )

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
    # DISTANCE FEATURES
    # --------------------------------------------------------

    data["distance_log"] = np.log1p(
        data["distance"].clip(lower=0)
    )

    data["distance_squared"] = (
        data["distance"] ** 2
    )

    # --------------------------------------------------------
    # WEIGHT FEATURES
    # --------------------------------------------------------

    data["weight_log"] = np.log1p(
        data["weight"].clip(lower=0)
    )

    data["weight_squared"] = (
        data["weight"] ** 2
    )

    safe_distance = (
        data["distance"]
        .replace(0, np.nan)
    )

    data["weight_per_mile"] = (
        data["weight"]
        / safe_distance
    )

    data["distance_weight_interaction"] = (
        data["distance"]
        * data["weight"]
    )

    # --------------------------------------------------------
    # MARKET FEATURES
    # --------------------------------------------------------

    data["market_index_squared"] = (
        data["market_index"] ** 2
    )

    data["market_index_log"] = np.log1p(
        data["market_index"].clip(lower=0)
    )

    # --------------------------------------------------------
    # QUOTE SIGNAL FEATURES
    # --------------------------------------------------------

    data["quote_signal_squared"] = (
        data["quote_signal"] ** 2
    )

    data["quote_signal_abs"] = (
        data["quote_signal"].abs()
    )

    # --------------------------------------------------------
    # GEOGRAPHICAL FEATURES
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


print("\nCreating features...")

train = create_features(
    train
)

validation = create_features(
    validation
)


# ============================================================
# PREPARE X / Y
# ============================================================

DROP_COLUMNS = [
    TARGET,
    "load_id",
    "date"
]

X_train = train.drop(
    columns=DROP_COLUMNS
)

y_train = train[TARGET]

X_validation = validation.drop(
    columns=[
        "load_id",
        "date"
    ]
)


# ============================================================
# FEATURE TYPES
# ============================================================

categorical_features = (
    X_train
    .select_dtypes(
        include=[
            "object",
            "category"
        ]
    )
    .columns
    .tolist()
)

numeric_features = (
    X_train
    .select_dtypes(
        include=[
            "number",
            "bool"
        ]
    )
    .columns
    .tolist()
)


print("\n" + "=" * 70)
print("FEATURE INFORMATION")
print("=" * 70)

print(
    "Numerical features:",
    len(numeric_features)
)

print(
    "Categorical features:",
    len(categorical_features)
)

print(
    "Total features before encoding:",
    len(
        numeric_features
        + categorical_features
    )
)


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
# FINAL MODEL
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
                **PARAMS
            )
        )
    ]
)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 70)
print("TRAINING FINAL XGBOOST")
print("=" * 70)

print(
    "\nTraining on ALL",
    len(X_train),
    "labeled rows..."
)

print(
    "\nTarget transformation: log1p(posted_rate)"
)

# ------------------------------------------------------------
# IMPORTANT
# ------------------------------------------------------------
# We train on log-transformed target because this was our
# best-performing validation strategy.

y_train_log = np.log1p(
    y_train
)

model.fit(
    X_train,
    y_train_log
)



MODEL_FILE = OUTPUT_DIR / "final_xgboost_model.pkl"

joblib.dump(
    model,
    MODEL_FILE
)

print(
    "\nModel saved to:",
    MODEL_FILE.resolve()
)


print(
    "\nFINAL MODEL TRAINING COMPLETE"
)


# ============================================================
# PREDICTION
# ============================================================

print("\n" + "=" * 70)
print("GENERATING FINAL PREDICTIONS")
print("=" * 70)

predictions_log = model.predict(
    X_validation
)

predictions = np.expm1(
    predictions_log
)

# Rates cannot be negative.
predictions = np.maximum(
    predictions,
    0
)


# ============================================================
# PREDICTION SANITY CHECK
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION SANITY CHECK")
print("=" * 70)

print(
    "Prediction count:",
    len(predictions)
)

print(
    "Expected count:",
    len(validation)
)

print(
    "Minimum:",
    f"${predictions.min():,.2f}"
)

print(
    "Maximum:",
    f"${predictions.max():,.2f}"
)

print(
    "Mean:",
    f"${predictions.mean():,.2f}"
)

print(
    "Median:",
    f"${np.median(predictions):,.2f}"
)

print(
    "Std:",
    f"${predictions.std():,.2f}"
)

print(
    "NaN predictions:",
    np.isnan(predictions).sum()
)

print(
    "Infinite predictions:",
    np.isinf(predictions).sum()
)


# ============================================================
# CREATE SUBMISSION
# ============================================================

submission = pd.DataFrame({
    "load_id": validation["load_id"],
    "predicted_rate": predictions
})


# ============================================================
# ROUNDING
# ============================================================

submission["predicted_rate"] = (
    submission["predicted_rate"]
    .round(2)
)


# ============================================================
# VERIFY SUBMISSION
# ============================================================

assert len(submission) == 12000

assert list(
    submission.columns
) == [
    "load_id",
    "predicted_rate"
]

assert submission["load_id"].equals(
    template["load_id"]
)

assert (
    submission["predicted_rate"]
    .notna()
    .all()
)

assert np.isfinite(
    submission["predicted_rate"]
).all()

assert (
    submission["predicted_rate"] >= 0
).all()


# ============================================================
# SAVE FINAL SUBMISSION
# ============================================================

submission_file = (
    OUTPUT_DIR
    / "final_predictions.csv"
)

submission.to_csv(
    submission_file,
    index=False
)


# ============================================================
# SAVE PREDICTION SUMMARY
# ============================================================

summary = pd.DataFrame({
    "metric": [
        "prediction_count",
        "minimum",
        "maximum",
        "mean",
        "median",
        "std",
        "p25",
        "p75",
        "p95",
        "p99"
    ],
    "value": [
        len(predictions),
        predictions.min(),
        predictions.max(),
        predictions.mean(),
        np.median(predictions),
        predictions.std(),
        np.percentile(
            predictions,
            25
        ),
        np.percentile(
            predictions,
            75
        ),
        np.percentile(
            predictions,
            95
        ),
        np.percentile(
            predictions,
            99
        )
    ]
})

summary.to_csv(
    OUTPUT_DIR
    / "prediction_summary.csv",
    index=False
)


# ============================================================
# DISPLAY SAMPLE
# ============================================================

print("\n" + "=" * 70)
print("FINAL PREDICTION SAMPLE")
print("=" * 70)

print(
    submission.head(20).to_string(
        index=False
    )
)


print("\n" + "=" * 70)
print("FINAL SUBMISSION CREATED")
print("=" * 70)

print(
    "\nFile:",
    submission_file.resolve()
)

print(
    "\nRows:",
    len(submission)
)

print(
    "\nColumns:",
    submission.columns.tolist()
)

print(
    "\n" + "=" * 70
)
print("STEP 16 COMPLETE")
print("=" * 70)