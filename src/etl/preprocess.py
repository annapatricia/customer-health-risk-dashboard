import pandas as pd

CATEGORICAL = ["payment_method", "contract"]
NUMERIC = [
    "tenure_months","monthly_charges","total_charges",
    "support_tickets_90d","late_payments_6m","has_auto_pay",
    "has_premium","usage_drop_30d","nps_score"
]
TARGET = "churn"

def load_data(path="data_sample/churn_sample.csv"):
    return pd.read_csv(path)

def one_hot(df: pd.DataFrame):
    return pd.get_dummies(df, columns=CATEGORICAL, drop_first=True)

def split_xy(df: pd.DataFrame):
    feature_cols = NUMERIC + [
        c for c in df.columns
        if c.startswith("payment_method_") or c.startswith("contract_")
    ]
    X = df[feature_cols].copy()
    y = df[TARGET].astype(int).copy()
    return X, y
