"""Core ML workflow for Sri Lanka Next-Day Rainfall Ensemble Dashboard."""
from pathlib import Path
import json
import urllib.request

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, ExtraTreesClassifier, StackingClassifier, VotingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

DATA_URL = "https://raw.githubusercontent.com/dhananjaya-hbc/lanka-data-tracker/main/data/weather.csv"
DATA_PATH = Path("data/weather.csv")
ARTIFACT_DIR = Path("artifacts")

NUM_FEATURES = [
    "temp_max_c", "temp_min_c", "precipitation_mm", "wind_speed_max_kmh", "humidity_mean_pct",
    "rain_lag1_mm", "rain_lag2_mm", "temp_max_lag1_c", "humidity_lag1_pct", "wind_lag1_kmh",
    "month_sin", "month_cos", "day_of_year_sin", "day_of_year_cos",
]
CAT_FEATURES = ["city"]
FEATURES = CAT_FEATURES + NUM_FEATURES
TARGET = "rain_tomorrow"


def download_dataset(force=False):
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists() and not force:
        return DATA_PATH
    print(f"Downloading dataset from: {DATA_URL}")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)
    print(f"Saved: {DATA_PATH}")
    return DATA_PATH



def load_and_prepare_data(path=None):
    path = Path(path) if path else DATA_PATH
    if not path.exists():
        download_dataset()
    df = pd.read_csv(path)
    expected = {"date", "city", "temp_max_c", "temp_min_c", "precipitation_mm", "wind_speed_max_kmh", "humidity_mean_pct"}
    missing = expected - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing expected columns: {sorted(missing)}. Check the source CSV schema.")
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date", "city"]).sort_values(["city", "date"]).copy()
    numeric = ["temp_max_c", "temp_min_c", "precipitation_mm", "wind_speed_max_kmh", "humidity_mean_pct"]
    for col in numeric:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    grouped = df.groupby("city", group_keys=False)
    df["rain_lag1_mm"] = grouped["precipitation_mm"].shift(1)
    df["rain_lag2_mm"] = grouped["precipitation_mm"].shift(2)
    df["temp_max_lag1_c"] = grouped["temp_max_c"].shift(1)
    df["humidity_lag1_pct"] = grouped["humidity_mean_pct"].shift(1)
    df["wind_lag1_kmh"] = grouped["wind_speed_max_kmh"].shift(1)
    df["month_sin"] = np.sin(2 * np.pi * df["date"].dt.month / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["date"].dt.month / 12)
    day = df["date"].dt.dayofyear
    df["day_of_year_sin"] = np.sin(2 * np.pi * day / 365.25)
    df["day_of_year_cos"] = np.cos(2 * np.pi * day / 365.25)
    df["target_date"] = grouped["date"].shift(-1)
    next_rain = grouped["precipitation_mm"].shift(-1)
    previous_date = grouped["date"].shift(1)
    two_days_previous_date = grouped["date"].shift(2)
    next_date_gap = (df["target_date"] - df["date"]).dt.days
    yesterday_gap = (df["date"] - previous_date).dt.days
    two_day_gap = (df["date"] - two_days_previous_date).dt.days
    df[TARGET] = (next_rain >= 1.0).astype(float)
    df.loc[next_rain.isna(), TARGET] = np.nan
    df.loc[next_date_gap != 1, TARGET] = np.nan
    df.loc[yesterday_gap != 1, ["rain_lag1_mm", "temp_max_lag1_c", "humidity_lag1_pct", "wind_lag1_kmh"]] = np.nan
    df.loc[two_day_gap != 2, "rain_lag2_mm"] = np.nan
    df = df.dropna(subset=FEATURES + [TARGET, "target_date"]).copy()
    df[TARGET] = df[TARGET].astype(int)
    return df


def build_models():
    def make_preprocessor():
        return ColumnTransformer([
            ("numeric", Pipeline([("imputer", SimpleImputer(strategy="median")), ("scaler", StandardScaler())]), NUM_FEATURES),
            ("city", Pipeline([("imputer", SimpleImputer(strategy="most_frequent")), ("encoder", OneHotEncoder(handle_unknown="ignore"))]), CAT_FEATURES),
        ])
    def pipeline(model):
        return Pipeline([("prep", make_preprocessor()), ("model", model)])
    bagging = pipeline(RandomForestClassifier(n_estimators=250, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1))
    boosting = pipeline(GradientBoostingClassifier(n_estimators=160, learning_rate=0.06, max_depth=2, random_state=42))
    extra = pipeline(ExtraTreesClassifier(n_estimators=200, min_samples_leaf=2, class_weight="balanced", random_state=42, n_jobs=-1))
    stack = StackingClassifier(
        estimators=[("rf", bagging), ("gb", boosting), ("et", extra)],
        final_estimator=LogisticRegression(max_iter=1000, class_weight="balanced"),
        stack_method="predict_proba", cv=3, n_jobs=1
    )
    hybrid = VotingClassifier(estimators=[("rf", bagging), ("gb", boosting)], voting="soft", weights=[1, 1])
    return {"Bagging - Random Forest": bagging, "Boosting - Gradient Boosting": boosting, "Stacking": stack, "Hybrid - Soft Voting": hybrid}



def chronological_split(df, test_fraction=0.2):
    cutoff = df["target_date"].quantile(1 - test_fraction)
    train = df[df["target_date"] < cutoff].copy()
    test = df[df["target_date"] >= cutoff].copy()
    return train, test, cutoff



def evaluate_models(df):
    train, test, cutoff = chronological_split(df)
    X_train, y_train = train[FEATURES], train[TARGET]
    X_test, y_test = test[FEATURES], test[TARGET]
    models = build_models()
    results = {}
    for name, model in models.items():
        print(f"Training: {name}")
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        results[name] = {
            "Accuracy": float(accuracy_score(y_test, pred)),
            "Balanced Accuracy": float(balanced_accuracy_score(y_test, pred)),
            "Precision": float(precision_score(y_test, pred, zero_division=0)),
            "Recall": float(recall_score(y_test, pred, zero_division=0)),
            "F1": float(f1_score(y_test, pred, zero_division=0)),
            "ROC-AUC": float(roc_auc_score(y_test, proba)) if proba is not None and y_test.nunique() > 1 else None,
        }
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    import joblib
    joblib.dump(models, ARTIFACT_DIR / "models.joblib")
    (ARTIFACT_DIR / "metrics.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    (ARTIFACT_DIR / "meta.json").write_text(json.dumps({"features": FEATURES, "numeric_features": NUM_FEATURES, "categorical_features": CAT_FEATURES, "threshold_mm": 1.0, "target": "Rain tomorrow (precipitation >= 1 mm)"}, indent=2), encoding="utf-8")
    return results, models, train, test, cutoff



def metrics_frame(metrics):
    return pd.DataFrame(metrics).T.sort_values("F1", ascending=False)
