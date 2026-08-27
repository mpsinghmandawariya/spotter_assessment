from pathlib import Path
import pandas as pd
import numpy as np


# ============================================================
# CONFIGURATION
# ============================================================

DATA_DIR = Path("data")

TRAIN_FILE = DATA_DIR / "train-test.csv"
VALIDATION_FILE = DATA_DIR / "validation.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("FREIGHT RATE PREDICTION - DATASET AUDIT")
print("=" * 70)

train = pd.read_csv(TRAIN_FILE)
validation = pd.read_csv(VALIDATION_FILE)


# ============================================================
# 1. BASIC DATASET INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("1. DATASET SHAPE")
print("=" * 70)

print(f"Training data shape    : {train.shape}")
print(f"Validation data shape  : {validation.shape}")


# ============================================================
# 2. COLUMN INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("2. COLUMN INFORMATION")
print("=" * 70)

print("\nTraining columns:")
for i, column in enumerate(train.columns, start=1):
    print(f"{i:2}. {column}")

print("\nValidation columns:")
for i, column in enumerate(validation.columns, start=1):
    print(f"{i:2}. {column}")


# ============================================================
# 3. DATA TYPES
# ============================================================

print("\n" + "=" * 70)
print("3. DATA TYPES")
print("=" * 70)

print(train.dtypes)


# ============================================================
# 4. FIRST 5 ROWS
# ============================================================

print("\n" + "=" * 70)
print("4. FIRST 5 TRAINING ROWS")
print("=" * 70)

print(train.head())


# ============================================================
# 5. LAST 5 ROWS
# ============================================================

print("\n" + "=" * 70)
print("5. LAST 5 TRAINING ROWS")
print("=" * 70)

print(train.tail())


# ============================================================
# 6. MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("6. MISSING VALUES")
print("=" * 70)

missing = pd.DataFrame({
    "column": train.columns,
    "missing_count": train.isna().sum().values,
    "missing_percentage": (
        train.isna().mean().values * 100
    )
})

missing = missing.sort_values(
    "missing_count",
    ascending=False
)

print(missing.to_string(index=False))


# ============================================================
# 7. DUPLICATES
# ============================================================

print("\n" + "=" * 70)
print("7. DUPLICATE ROWS")
print("=" * 70)

duplicate_count = train.duplicated().sum()

print(f"Duplicate rows: {duplicate_count}")


# ============================================================
# 8. UNIQUE VALUES
# ============================================================

print("\n" + "=" * 70)
print("8. UNIQUE VALUES")
print("=" * 70)

unique_info = pd.DataFrame({
    "column": train.columns,
    "unique_values": [
        train[col].nunique(dropna=False)
        for col in train.columns
    ]
})

print(unique_info.to_string(index=False))


# ============================================================
# 9. NUMERICAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("9. NUMERICAL FEATURES SUMMARY")
print("=" * 70)

numeric_columns = train.select_dtypes(
    include=np.number
).columns

print(
    train[numeric_columns]
    .describe()
    .T
    .to_string()
)


# ============================================================
# 10. CATEGORICAL FEATURES
# ============================================================

print("\n" + "=" * 70)
print("10. CATEGORICAL FEATURES")
print("=" * 70)

categorical_columns = train.select_dtypes(
    include=["object", "category"]
).columns

for column in categorical_columns:

    print(f"\n--- {column} ---")

    print(
        train[column]
        .value_counts(dropna=False)
        .head(20)
        .to_string()
    )


# ============================================================
# 11. DATE INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("11. DATE INFORMATION")
print("=" * 70)

if "date" in train.columns:

    train["date"] = pd.to_datetime(
        train["date"],
        errors="coerce"
    )

    validation["date"] = pd.to_datetime(
        validation["date"],
        errors="coerce"
    )

    print(
        f"Training date range   : "
        f"{train['date'].min()} → {train['date'].max()}"
    )

    print(
        f"Validation date range : "
        f"{validation['date'].min()} → {validation['date'].max()}"
    )

    print(
        f"Invalid training dates : "
        f"{train['date'].isna().sum()}"
    )

    print(
        f"Invalid validation dates : "
        f"{validation['date'].isna().sum()}"
    )


# ============================================================
# 12. TARGET ANALYSIS
# ============================================================

TARGET = "posted_rate"

print("\n" + "=" * 70)
print(f"12. TARGET ANALYSIS: {TARGET}")
print("=" * 70)

if TARGET in train.columns:

    target = pd.to_numeric(
        train[TARGET],
        errors="coerce"
    )

    print(f"Missing target values : {target.isna().sum()}")
    print(f"Minimum               : {target.min():.2f}")
    print(f"Maximum               : {target.max():.2f}")
    print(f"Mean                  : {target.mean():.2f}")
    print(f"Median                : {target.median():.2f}")
    print(f"Std deviation         : {target.std():.2f}")

    print("\nTarget percentiles:")

    percentiles = target.quantile(
        [0.01, 0.05, 0.10, 0.25, 0.50,
         0.75, 0.90, 0.95, 0.99]
    )

    print(percentiles.to_string())


# ============================================================
# 13. CHECK NUMERICAL VALUES
# ============================================================

print("\n" + "=" * 70)
print("13. NON-FINITE NUMERICAL VALUES")
print("=" * 70)

for column in numeric_columns:

    values = pd.to_numeric(
        train[column],
        errors="coerce"
    )

    infinite_count = np.isinf(values).sum()

    if infinite_count > 0:
        print(
            f"{column}: {infinite_count} infinite values"
        )


# ============================================================
# 14. POSSIBLE NEGATIVE VALUES
# ============================================================

print("\n" + "=" * 70)
print("14. NEGATIVE NUMERICAL VALUES")
print("=" * 70)

for column in numeric_columns:

    values = pd.to_numeric(
        train[column],
        errors="coerce"
    )

    negative_count = (values < 0).sum()

    if negative_count > 0:
        print(
            f"{column}: {negative_count} negative values"
        )


# ============================================================
# 15. TARGET OUTLIER CHECK
# ============================================================

print("\n" + "=" * 70)
print("15. TARGET OUTLIER CHECK")
print("=" * 70)

if TARGET in train.columns:

    q1 = target.quantile(0.25)
    q3 = target.quantile(0.75)

    iqr = q3 - q1

    lower_bound = q1 - 1.5 * iqr
    upper_bound = q3 + 1.5 * iqr

    outliers = train[
        (target < lower_bound) |
        (target > upper_bound)
    ]

    print(f"Q1             : {q1:.2f}")
    print(f"Q3             : {q3:.2f}")
    print(f"IQR            : {iqr:.2f}")
    print(f"Lower bound    : {lower_bound:.2f}")
    print(f"Upper bound    : {upper_bound:.2f}")
    print(f"Outlier count   : {len(outliers)}")
    print(
        f"Outlier %       : "
        f"{len(outliers) / len(train) * 100:.2f}%"
    )


# ============================================================
# 16. VALIDATION DATA CHECK
# ============================================================

print("\n" + "=" * 70)
print("16. VALIDATION DATA CHECK")
print("=" * 70)

print("Validation shape:", validation.shape)

print("\nValidation missing values:")

validation_missing = validation.isna().sum()

print(
    validation_missing[
        validation_missing > 0
    ].to_string()
    if (validation_missing > 0).any()
    else "No missing values found."
)


# ============================================================
# 17. LOAD ID CHECK
# ============================================================

print("\n" + "=" * 70)
print("17. LOAD ID CHECK")
print("=" * 70)

if "load_id" in validation.columns:

    print(
        "Unique validation load IDs:",
        validation["load_id"].nunique()
    )

    print(
        "Duplicate validation IDs:",
        validation["load_id"].duplicated().sum()
    )

    print("\nFirst 10 IDs:")

    print(
        validation["load_id"]
        .head(10)
        .to_string(index=False)
    )


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 70)
print("DATASET AUDIT COMPLETE")
print("=" * 70)
