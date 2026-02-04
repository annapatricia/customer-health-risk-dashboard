import numpy as np
import pandas as pd

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def main(out_path="data_sample/churn_sample.csv", n=5000, seed=42):
    rng = np.random.default_rng(seed)

    tenure_months = rng.integers(1, 61, size=n)
    monthly_charges = rng.normal(110, 35, size=n).clip(20, 300)
    total_charges = (monthly_charges * tenure_months + rng.normal(0, 50, size=n)).clip(0, None)

    support_tickets_90d = rng.poisson(1.2, size=n)
    late_payments_6m = rng.poisson(0.6, size=n)
    payment_method = rng.choice(["credit_card", "boleto", "pix"], size=n, p=[0.55, 0.25, 0.20])

    contract = rng.choice(["month_to_month", "one_year", "two_year"], size=n, p=[0.60, 0.25, 0.15])
    has_auto_pay = rng.choice([0, 1], size=n, p=[0.55, 0.45])
    has_premium = rng.choice([0, 1], size=n, p=[0.65, 0.35])

    usage_drop_30d = rng.normal(0.05, 0.18, size=n)  # positivo = caiu
    nps_score = rng.normal(20, 35, size=n).clip(-100, 100)

    z = (
        1.1 * (contract == "month_to_month").astype(int)
        + 0.7 * (payment_method == "boleto").astype(int)
        + 0.6 * (has_auto_pay == 0).astype(int)
        + 0.35 * support_tickets_90d
        + 0.45 * late_payments_6m
        + 1.6 * usage_drop_30d
        - 0.02 * tenure_months
        - 0.015 * (nps_score)
        + rng.normal(0, 0.6, size=n)
    )

    p = sigmoid(z)
    churn = (rng.random(n) < p).astype(int)

    df = pd.DataFrame({
        "customer_id": [f"C{str(i).zfill(6)}" for i in range(n)],
        "tenure_months": tenure_months,
        "monthly_charges": monthly_charges.round(2),
        "total_charges": total_charges.round(2),
        "support_tickets_90d": support_tickets_90d,
        "late_payments_6m": late_payments_6m,
        "payment_method": payment_method,
        "contract": contract,
        "has_auto_pay": has_auto_pay,
        "has_premium": has_premium,
        "usage_drop_30d": usage_drop_30d.round(3),
        "nps_score": nps_score.round(0).astype(int),
        "churn": churn
    })

    df.to_csv(out_path, index=False)
    print(f"Saved: {out_path} ({len(df)} rows)")

if __name__ == "__main__":
    main()
