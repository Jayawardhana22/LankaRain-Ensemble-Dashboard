from ml_core import load_and_prepare_data, evaluate_models, metrics_frame


def main():
    print("Loading Sri Lankan weather data and engineering features...")
    df = load_and_prepare_data()
    print(f"Prepared rows: {len(df):,} | Cities: {df['city'].nunique()} | Target rain rate: {df['rain_tomorrow'].mean():.1%}")
    metrics, models, train, test, cutoff = evaluate_models(df)
    print("\nTest set begins at target date:", cutoff.date())
    print("\nMODEL COMPARISON (chronological test split)")
    print(metrics_frame(metrics).round(3).to_string())
    print("\nSaved artifacts/models.joblib, artifacts/metrics.json, and artifacts/meta.json")
    print("Run the dashboard with: streamlit run app.py")

if __name__ == "__main__":
    main()
