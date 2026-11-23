
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from ml_core import (
    load_and_prepare_data,
    evaluate_models,
    FEATURES,
    NUM_FEATURES,
    metrics_frame,
)

st.set_page_config(
    page_title="LankaRain Ensemble Lab",
    page_icon="🌧️",
    layout="wide",
)

ARTIFACT_DIR = Path("artifacts")
MODEL_PATH = ARTIFACT_DIR / "models.joblib"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


@st.cache_data(show_spinner="Loading Sri Lankan weather dataset...")
def get_data():
    return load_and_prepare_data()


@st.cache_resource(show_spinner="Loading or training ensemble models...")
def get_models():
    if MODEL_PATH.exists() and METRICS_PATH.exists():
        models = joblib.load(MODEL_PATH)
        metrics = json.loads(
            METRICS_PATH.read_text(encoding="utf-8")
        )
    else:
        st.info(
            "Trained models were not found. "
            "Training the ensemble models for the first time. "
            "Please wait..."
        )

        df = get_data()
        metrics, models, train, test, cutoff = evaluate_models(df)

    return models, metrics


st.title("🌧️ LankaRain Ensemble Lab")
st.caption(
    "Sri Lanka next-day rainfall classification | "
    "Bagging · Boosting · Stacking · Hybrid voting"
)

# Load dataset
try:
    df = get_data()
except Exception as e:
    st.error(f"Could not load the weather dataset: {e}")
    st.stop()

# Load or train models
try:
    models, metrics = get_models()
except Exception as e:
    st.error(f"Could not load or train the models: {e}")
    st.exception(e)
    st.stop()


# Sidebar controls
st.sidebar.header("Prediction controls")

city_options = sorted(df["city"].unique().tolist())
city = st.sidebar.selectbox("Sri Lankan city", city_options)

city_data = df[df["city"] == city]
city_medians = city_data[NUM_FEATURES].median(numeric_only=True)


def slider(label, key, minimum, maximum, step, unit="", value=None):
    val = (
        float(city_medians.get(key, 0.0))
        if value is None
        else float(value)
    )
    val = min(max(val, minimum), maximum)

    format_unit = unit.replace("%", "%%")

    return st.sidebar.slider(
        label,
        min_value=float(minimum),
        max_value=float(maximum),
        value=val,
        step=float(step),
        format=f"%.1f {format_unit}" if unit else "%.1f",
    )


current = {
    "temp_max_c": slider(
        "Today's max temperature (°C)",
        "temp_max_c", 10, 45, 0.5, "°C"
    ),
    "temp_min_c": slider(
        "Today's min temperature (°C)",
        "temp_min_c", 5, 35, 0.5, "°C"
    ),
    "precipitation_mm": slider(
        "Today's precipitation (mm)",
        "precipitation_mm", 0, 250, 1, "mm"
    ),
    "wind_speed_max_kmh": slider(
        "Today's max wind (km/h)",
        "wind_speed_max_kmh", 0, 150, 1, "km/h"
    ),
    "humidity_mean_pct": slider(
        "Today's mean humidity (%)",
        "humidity_mean_pct", 0, 100, 1, "%"
    ),
    "rain_lag1_mm": slider(
        "Rain yesterday (mm)",
        "rain_lag1_mm", 0, 250, 1, "mm"
    ),
    "rain_lag2_mm": slider(
        "Rain 2 days ago (mm)",
        "rain_lag2_mm", 0, 250, 1, "mm"
    ),
    "temp_max_lag1_c": slider(
        "Yesterday max temperature (°C)",
        "temp_max_lag1_c", 10, 45, 0.5, "°C"
    ),
    "humidity_lag1_pct": slider(
        "Yesterday humidity (%)",
        "humidity_lag1_pct", 0, 100, 1, "%"
    ),
    "wind_lag1_kmh": slider(
        "Yesterday max wind (km/h)",
        "wind_lag1_kmh", 0, 150, 1, "km/h"
    ),
}

# Seasonal inputs
month = st.sidebar.select_slider(
    "Month",
    options=list(range(1, 13)),
    value=int(pd.Timestamp.today().month),
)

day_of_year = st.sidebar.slider(
    "Day of year",
    min_value=1,
    max_value=365,
    value=min(int(pd.Timestamp.today().dayofyear), 365),
    step=1,
)

current["month_sin"] = np.sin(2 * np.pi * month / 12)
current["month_cos"] = np.cos(2 * np.pi * month / 12)
current["day_of_year_sin"] = np.sin(
    2 * np.pi * day_of_year / 365.25
)
current["day_of_year_cos"] = np.cos(
    2 * np.pi * day_of_year / 365.25
)

row = pd.DataFrame([{**current, "city": city}])[FEATURES]

model_names = list(models.keys())

default_index = (
    model_names.index("Hybrid - Soft Voting")
    if "Hybrid - Soft Voting" in model_names
    else 0
)

selected_model_name = st.selectbox(
    "Model for prediction",
    model_names,
    index=default_index,
)

model = models[selected_model_name]

prob = float(model.predict_proba(row)[0, 1])
prediction = int(prob >= 0.5)


# Overview cards
st.subheader("Dataset overview")

c1, c2, c3, c4 = st.columns(4)

c1.metric("Records after preparation", f"{len(df):,}")
c2.metric("Sri Lankan cities", f"{df.city.nunique()}")
c3.metric("Rain tomorrow rate", f"{df.rain_tomorrow.mean():.1%}")
c4.metric("Selected model", selected_model_name)


# Prediction
st.subheader("Rainfall prediction")

left, right = st.columns(2)

with left:
    if prediction:
        st.error(
            f"Rain likely tomorrow — probability: {prob:.1%}"
        )
    else:
        st.success(
            f"Rain not likely tomorrow — probability: {prob:.1%}"
        )

    st.progress(prob)

with right:
    st.write("**Rain probability**")

    fig, ax = plt.subplots()
    ax.bar(["No rain", "Rain"], [1 - prob, prob])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Probability")

    st.pyplot(fig, clear_figure=True)


# Model comparison
st.subheader("Ensemble model comparison")

metrics_df = metrics_frame(metrics)
fmt_df = metrics_df.copy()

for col in fmt_df.columns:
    fmt_df[col] = fmt_df[col].apply(
        lambda x: "—" if pd.isna(x) else f"{x:.3f}"
    )

st.dataframe(fmt_df, use_container_width=True)

metric_choice = st.selectbox(
    "Metric to visualize",
    [
        "F1",
        "Balanced Accuracy",
        "Recall",
        "Precision",
        "Accuracy",
        "ROC-AUC",
    ],
)

plot_metrics = (
    pd.DataFrame(metrics)
    .T
    .sort_values(metric_choice, ascending=False)
)

fig, ax = plt.subplots()
ax.bar(plot_metrics.index, plot_metrics[metric_choice])
ax.set_ylim(0, 1)
ax.set_ylabel(metric_choice)
ax.set_title(f"Ensemble performance — {metric_choice}")

plt.xticks(rotation=15, ha="right")
plt.tight_layout()

st.pyplot(fig, clear_figure=True)


# Rainfall history
st.subheader(f"Rainfall history — {city}")

city_history = (
    city_data.sort_values("date")
    .tail(365)
)

fig, ax = plt.subplots(figsize=(10, 3.5))
ax.plot(
    city_history["date"],
    city_history["precipitation_mm"],
)

ax.set_ylabel("Rainfall (mm)")
ax.set_xlabel("Date")
ax.set_title("Daily precipitation (last 365 available records)")

plt.tight_layout()
st.pyplot(fig, clear_figure=True)


# Feature importance
st.subheader("Feature influence — Random Forest")

rf = models["Bagging - Random Forest"]

try:
    names = rf.named_steps["prep"].get_feature_names_out()
    importance = rf.named_steps["model"].feature_importances_

    imp_df = (
        pd.DataFrame({
            "Feature": names,
            "Importance": importance,
        })
        .sort_values("Importance", ascending=False)
        .head(12)
    )

    fig, ax = plt.subplots()

    ax.barh(
        imp_df["Feature"][::-1],
        imp_df["Importance"][::-1],
    )

    ax.set_xlabel("Feature importance")
    ax.set_title("Top 12 influential features")

    plt.tight_layout()
    st.pyplot(fig, clear_figure=True)

    st.write(
        "Feature importance shows which inputs the trained "
        "Random Forest relied on most. It is not causal proof."
    )

except Exception as e:
    st.info(f"Feature importance unavailable: {e}")


# Experiment information
st.subheader("About the experiment")

st.markdown("""
**Target:** Rain tomorrow is positive when the next day's
recorded precipitation is at least 1 mm.

**Features:** Current-day weather, two rain lags,
one-day temperature/humidity/wind lags, city,
and seasonal calendar features.

**Evaluation:** Chronological split (latest 20% of target
dates held out).

**Models:** Random Forest (Bagging), Gradient Boosting,
Stacking, and a soft-voting hybrid.

**Caution:** This is a historical-pattern classifier,
not a meteorological forecast system.
""")

with st.expander("Dataset source and AI-use statement"):
    st.markdown(
        "Dataset: lanka-data-tracker (weather.csv), sourced "
        "from Open-Meteo and maintained as a Sri Lankan "
        "multi-city weather dataset. See project README for the link."
    )

    st.markdown(
        "**AI use disclosure:** AI tools were used to assist "
        "with code generation, debugging, and dashboard layout. "
        "The student reviewed, ran, and interpreted the "
        "implementation and results."
    )
