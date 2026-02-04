import os
import joblib
import numpy as np
import pandas as pd

from src.etl.preprocess import load_data, one_hot

def top_drivers_for_row(x_row, feature_names, coefs, topk=3):
    contrib = x_row.values * coefs
    idx = np.argsort(contrib)[::-1]

    drivers = []
    for i in idx:
        if contrib[i] > 0:
            drivers.append(f"{feature_names[i]}(+{contrib[i]:.2f})")
        if len(drivers) >= topk:
            break
    return "; ".join(drivers) if drivers else "no_positive_driver"

def main(
    model_path="outputs/model.joblib",
    risk_path="outputs/customer_risk_daily.csv",
    out_path="outputs/customer_risk_with_drivers.csv"
):
    print("Running explain...")

    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    if not os.path.exists(risk_path):
        raise FileNotFoundError(f"Risk file not found: {risk_path}")

    bundle = joblib.load(model_path)
    pipe = bundle["model"]
    cols = bundle["columns"]

    clf = pipe.named_steps["clf"]
    coefs = clf.coef_[0]

    df = load_data()
    df_hot = one_hot(df)

    for c in cols:
        if c not in df_hot.columns:
            df_hot[c] = 0
    df_hot = df_hot[cols]

    drivers = df_hot.apply(lambda r: top_drivers_for_row(r, cols, coefs, topk=3), axis=1)

    risk = pd.read_csv(risk_path)
    out = risk.merge(
        df[[
            "customer_id",
            "tenure_months",
            "support_tickets_90d",
            "late_payments_6m",
            "usage_drop_30d",
            "nps_score",
            "contract",
            "payment_method"
        ]],
        on="customer_id",
        how="left"
    )

    out["top_drivers"] = drivers.values

    os.makedirs("outputs", exist_ok=True)
    out.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(out)} rows)")

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print("ERROR:", e)
        raise

