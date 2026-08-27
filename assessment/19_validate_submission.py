from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# CONFIG
# ============================================================

SUBMISSION_FILE = Path(
    "outputs/final_model/final_predictions.csv"
)

VALIDATION_FILE = Path(
    "data/validation.csv"
)

TEMPLATE_FILE = Path(
    "data/validation-predictions-template.csv"
)

OUTPUT_DIR = Path(
    "outputs/final_model"
)


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("STEP 17 - FINAL SUBMISSION VALIDATION")
print("=" * 70)


# ============================================================
# LOAD
# ============================================================

submission = pd.read_csv(
    SUBMISSION_FILE
)

validation = pd.read_csv(
    VALIDATION_FILE
)

template = pd.read_csv(
    TEMPLATE_FILE
)


# ============================================================
# BASIC INFORMATION
# ============================================================

print("\n" + "=" * 70)
print("1. BASIC VALIDATION")
print("=" * 70)

print(
    "Submission shape:",
    submission.shape
)

print(
    "Expected shape:",
    template.shape
)

print(
    "Submission columns:",
    submission.columns.tolist()
)

print(
    "Expected columns:",
    template.columns.tolist()
)


# ============================================================
# COLUMN CHECK
# ============================================================

assert list(
    submission.columns
) == [
    "load_id",
    "predicted_rate"
]

print(
    "\n✓ Column names are correct"
)


# ============================================================
# ROW COUNT
# ============================================================

assert len(submission) == 12000

assert len(submission) == len(
    validation
)

assert len(submission) == len(
    template
)

print(
    "✓ Row count = 12,000"
)


# ============================================================
# LOAD ID CHECK
# ============================================================

print("\n" + "=" * 70)
print("2. LOAD ID VALIDATION")
print("=" * 70)

assert submission["load_id"].equals(
    validation["load_id"]
)

assert submission["load_id"].equals(
    template["load_id"]
)

print(
    "✓ Load IDs match validation.csv"
)

print(
    "✓ Load IDs match template"
)


# ============================================================
# DUPLICATE CHECK
# ============================================================

duplicate_ids = (
    submission["load_id"]
    .duplicated()
    .sum()
)

print(
    "Duplicate load IDs:",
    duplicate_ids
)

assert duplicate_ids == 0

print(
    "✓ No duplicate load IDs"
)


# ============================================================
# MISSING VALUES
# ============================================================

print("\n" + "=" * 70)
print("3. MISSING VALUE CHECK")
print("=" * 70)

missing_ids = (
    submission["load_id"]
    .isna()
    .sum()
)

missing_predictions = (
    submission["predicted_rate"]
    .isna()
    .sum()
)

print(
    "Missing load IDs:",
    missing_ids
)

print(
    "Missing predictions:",
    missing_predictions
)

assert missing_ids == 0

assert missing_predictions == 0

print(
    "✓ No missing values"
)


# ============================================================
# INFINITE VALUES
# ============================================================

print("\n" + "=" * 70)
print("4. NUMERICAL VALIDATION")
print("=" * 70)

infinite_predictions = np.isinf(
    submission["predicted_rate"]
).sum()

print(
    "Infinite predictions:",
    infinite_predictions
)

assert infinite_predictions == 0

print(
    "✓ No infinite predictions"
)


# ============================================================
# NEGATIVE VALUES
# ============================================================

negative_predictions = (
    submission["predicted_rate"] < 0
).sum()

print(
    "Negative predictions:",
    negative_predictions
)

assert negative_predictions == 0

print(
    "✓ No negative predictions"
)


# ============================================================
# ZERO VALUES
# ============================================================

zero_predictions = (
    submission["predicted_rate"] == 0
).sum()

print(
    "Zero predictions:",
    zero_predictions
)


# ============================================================
# DISTRIBUTION
# ============================================================

print("\n" + "=" * 70)
print("5. PREDICTION DISTRIBUTION")
print("=" * 70)

pred = submission[
    "predicted_rate"
]

print(
    f"Minimum : ${pred.min():,.2f}"
)

print(
    f"Maximum : ${pred.max():,.2f}"
)

print(
    f"Mean    : ${pred.mean():,.2f}"
)

print(
    f"Median  : ${pred.median():,.2f}"
)

print(
    f"Std     : ${pred.std():,.2f}"
)

print(
    f"P25     : ${pred.quantile(0.25):,.2f}"
)

print(
    f"P75     : ${pred.quantile(0.75):,.2f}"
)

print(
    f"P90     : ${pred.quantile(0.90):,.2f}"
)

print(
    f"P95     : ${pred.quantile(0.95):,.2f}"
)

print(
    f"P99     : ${pred.quantile(0.99):,.2f}"
)


# ============================================================
# RATE BUCKETS
# ============================================================

print("\n" + "=" * 70)
print("6. PREDICTION BUCKETS")
print("=" * 70)

buckets = pd.cut(
    pred,
    bins=[
        0,
        500,
        1000,
        2000,
        3000,
        5000,
        7500,
        10000,
        np.inf
    ],
    labels=[
        "<$500",
        "$500-$1k",
        "$1k-$2k",
        "$2k-$3k",
        "$3k-$5k",
        "$5k-$7.5k",
        "$7.5k-$10k",
        "$10k+"
    ],
    include_lowest=True
)

bucket_counts = (
    buckets
    .value_counts(
        sort=False
    )
    .reset_index()
)

bucket_counts.columns = [
    "rate_bucket",
    "count"
]

bucket_counts["percentage"] = (
    bucket_counts["count"]
    /
    len(submission)
    *
    100
)

print(
    bucket_counts.to_string(
        index=False
    )
)


# ============================================================
# TOP PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("7. TOP 30 PREDICTIONS")
print("=" * 70)

top_predictions = (
    submission
    .sort_values(
        "predicted_rate",
        ascending=False
    )
    .head(30)
)

print(
    top_predictions.to_string(
        index=False
    )
)


# ============================================================
# LOWEST PREDICTIONS
# ============================================================

print("\n" + "=" * 70)
print("8. LOWEST 20 PREDICTIONS")
print("=" * 70)

lowest_predictions = (
    submission
    .sort_values(
        "predicted_rate"
    )
    .head(20)
)

print(
    lowest_predictions.to_string(
        index=False
    )
)


# ============================================================
# TEMPLATE COMPATIBILITY
# ============================================================

print("\n" + "=" * 70)
print("9. TEMPLATE COMPATIBILITY")
print("=" * 70)

assert submission["load_id"].tolist() == (
    template["load_id"].tolist()
)

print(
    "✓ Submission load order exactly matches template"
)

print(
    "✓ Submission schema exactly matches template"
)


# ============================================================
# SAVE AUDIT
# ============================================================

audit = pd.DataFrame({
    "metric": [
        "rows",
        "columns",
        "missing_predictions",
        "infinite_predictions",
        "negative_predictions",
        "duplicate_load_ids",
        "minimum_prediction",
        "maximum_prediction",
        "mean_prediction",
        "median_prediction",
        "p95_prediction",
        "p99_prediction"
    ],
    "value": [
        len(submission),
        len(submission.columns),
        missing_predictions,
        infinite_predictions,
        negative_predictions,
        duplicate_ids,
        pred.min(),
        pred.max(),
        pred.mean(),
        pred.median(),
        pred.quantile(0.95),
        pred.quantile(0.99)
    ]
})

audit_file = (
    OUTPUT_DIR
    / "submission_audit.csv"
)

audit.to_csv(
    audit_file,
    index=False
)


# ============================================================
# FINAL STATUS
# ============================================================

print("\n" + "=" * 70)
print("FINAL SUBMISSION STATUS")
print("=" * 70)

print(
    "\n✓ 12,000 predictions"
)

print(
    "✓ Correct columns"
)

print(
    "✓ Correct load IDs"
)

print(
    "✓ Correct row order"
)

print(
    "✓ No missing predictions"
)

print(
    "✓ No infinite values"
)

print(
    "✓ No negative predictions"
)

print(
    "\nSubmission is READY."
)

print(
    "\nSubmission file:"
)

print(
    SUBMISSION_FILE.resolve()
)

print(
    "\nAudit file:"
)

print(
    audit_file.resolve()
)

print(
    "\n" + "=" * 70
)
print("STEP 17 COMPLETE")
print("=" * 70)