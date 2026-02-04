import os
import pandas as pd

def main(
    in_path="outputs/customer_risk_with_drivers.csv",
    out_path="outputs/mart_churn_dashboard.csv"
):
    if not os.path.exists(in_path):
        raise FileNotFoundError(
            f"Missing input file: {in_path}. Run: python -m src.model.explain"
        )

    df = pd.read_csv(in_path)

    # Score em 0-100 (mais amigável pro BI)
    df["risk_score"] = (df["churn_probability"] * 100).round(1)

    # Flags úteis para KPIs
    df["is_high_risk"] = (df["risk_level"] == "high").astype(int)
    df["is_medium_risk"] = (df["risk_level"] == "medium").astype(int)
    df["is_low_risk"] = (df["risk_level"] == "low").astype(int)

    # Categorias simples para BI (ajuda segmentação)
    df["tenure_bucket"] = pd.cut(
        df["tenure_months"],
        bins=[0, 6, 12, 24, 60],
        labels=["0-6", "7-12", "13-24", "25-60"],
        include_lowest=True
    )

    df["nps_bucket"] = pd.cut(
        df["nps_score"],
        bins=[-101, 0, 50, 100],
        labels=["detractor", "neutral", "promoter"],
        include_lowest=True
    )

    df["usage_drop_bucket"] = pd.cut(
        df["usage_drop_30d"],
        bins=[-10, 0.0, 0.10, 10],
        labels=["no_drop_or_increase", "small_drop", "large_drop"],
        include_lowest=True
    )

    os.makedirs("outputs", exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    main()
