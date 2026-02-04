import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

from src.etl.preprocess import load_data, one_hot, split_xy

def main(model_path="outputs/model.joblib", metrics_path="outputs/metrics.txt"):
    df = load_data()
    df = one_hot(df)

    X, y = split_xy(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    pipe = Pipeline([
        ("scaler", StandardScaler(with_mean=False)),
        ("clf", LogisticRegression(max_iter=2000))
    ])

    pipe.fit(X_train, y_train)

    proba = pipe.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    report = classification_report(y_test, (proba >= 0.5).astype(int))

    os.makedirs("outputs", exist_ok=True)

    joblib.dump({"model": pipe, "columns": X.columns.tolist()}, model_path)

    text = f"AUC: {auc:.4f}\n\n{report}\n"
    with open(metrics_path, "w", encoding="utf-8") as f:
        f.write(text)

    print(text)
    print(f"Saved model: {model_path}")
    print(f"Saved metrics: {metrics_path}")

if __name__ == "__main__":
    main()
