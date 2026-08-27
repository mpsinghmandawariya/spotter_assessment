from pathlib import Path

import numpy as np
import pandas as pd
import joblib


# ============================================================
# CONFIG
# ============================================================

MODEL_FILE = Path(
    "outputs/final_model/final_xgboost_model.pkl"
)


# ============================================================
# LOAD MODEL
# ============================================================

print("=" * 70)
print("CUSTOM FREIGHT RATE PREDICTION")
print("=" * 70)

model = joblib.load(
    MODEL_FILE
)


# ============================================================
# USER INPUT
# ============================================================

print("\nEnter shipment details.\n")


pickup = input(
    "Pickup city: "
).strip()

delivery = input(
    "Delivery city: "
).strip()

equipment = input(
    "Equipment (Dry Van / Reefer / Flatbed): "
).strip()

distance = float(
    input(
        "Distance (miles): "
    )
)

weight = float(
    input(
        "Weight (lbs): "
    )
)

pickup_lat = float(
    input(
        "Pickup latitude: "
    )
)

pickup_lon = float(
    input(
        "Pickup longitude: "
    )
)

delivery_lat = float(
    input(
        "Delivery latitude: "
    )
)

delivery_lon = float(
    input(
        "Delivery longitude: "
    )
)

date = input(
    "Date (YYYY-MM-DD): "
).strip()

market_index = float(
    input(
        "Market index: "
    )
)

quote_signal = float(
    input(
        "Quote signal: "
    )
)


# ============================================================
# CREATE DATAFRAME
# ============================================================

data = pd.DataFrame([
    {
        "load_id": "CUSTOM-001",
        "pickup": pickup,
        "delivery": delivery,
        "pickup_lat": pickup_lat,
        "pickup_lon": pickup_lon,
        "delivery_lat": delivery_lat,
        "delivery_lon": delivery_lon,
        "distance": distance,
        "equipment": equipment,
        "weight": weight,
        "date": date,
        "market_index": market_index,
        "quote_signal": quote_signal
    }
])


data["date"] = pd.to_datetime(
    data["date"],
    errors="coerce"
)


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(data):

    data = data.copy()

    # --------------------------------------------------------
    # DATE
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
    # CYCLIC DATE
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
    # MARKET
    # --------------------------------------------------------

    data["market_index_squared"] = (
        data["market_index"] ** 2
    )

    data["market_index_log"] = np.log1p(
        data["market_index"].clip(lower=0)
    )

    # --------------------------------------------------------
    # QUOTE
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


data = create_features(
    data
)


# ============================================================
# REMOVE NON-FEATURE COLUMNS
# ============================================================

X = data.drop(
    columns=[
        "load_id",
        "date"
    ]
)


# ============================================================
# PREDICT
# ============================================================

prediction_log = model.predict(
    X
)

prediction = np.expm1(
    prediction_log
)[0]

prediction = max(
    prediction,
    0
)


# ============================================================
# RESULT
# ============================================================

print("\n" + "=" * 70)
print("PREDICTION RESULT")
print("=" * 70)

print(
    f"\nRoute       : {pickup} → {delivery}"
)

print(
    f"Equipment   : {equipment}"
)

print(
    f"Distance    : {distance:,.1f} miles"
)

print(
    f"Weight      : {weight:,.0f} lbs"
)

print(
    f"Date        : {date}"
)

print(
    f"\nPredicted Freight Rate: ${prediction:,.2f}"
)

print(
    "\n" + "=" * 70
)