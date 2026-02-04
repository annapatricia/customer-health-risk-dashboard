import os
import joblib
import pandas as pd
from src.etl.preprocess import load_data, one_hot

def risk_level(p: float) -> str:
    if p >= 0.75:
        return "high"
    if p >= 0.50:
        return "medium"
    return "low"

def main(model_path="outputs/model.joblib", out_path="outputs/customer_risk_daily.csv"):
    bundle = joblib.load(model_path)
    model = bundle["model"]
    cols = bundle["columns"]

    df = load_data()
    df_hot = one_hot(df)

    # garantir mesmas colunas do treino
    for c in cols:
        if c not in df_hot.columns:
            df_hot[c] = 0
    df_hot = df_hot[cols]

    proba = model.predict_proba(df_hot)[:, 1]

    out = df[["customer_id"]].copy()
    out["churn_probability"] = proba.round(4)
    out["risk_level"] = out["churn_probability"].apply(risk_level)

    os.makedirs("outputs", exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(out)} rows)")

if __name__ == "__main__":
    main()
