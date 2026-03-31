from utils import LABEL_NAMES, OUTPUT_DIR, SEED

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_rfc(X: pd.DataFrame, y: pd.Series) -> RandomForestClassifier:
    rfc = RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    rfc.fit(X, y)
    return rfc

def evaluate_rfc(rfc: RandomForestClassifier, X: pd.DataFrame, y: pd.Series) -> None:
    y_pred = rfc.predict(X)
    print("\nClassification report (train):")
    print(classification_report(y, y_pred, target_names=list(LABEL_NAMES.values())))

    importances = (
        pd.Series(rfc.feature_importances_, index=X.columns)
        .sort_values(ascending=False)
        .head(15)
    )
    print("\nTop 15 feature importances:")
    print(importances.to_string())
    importances.to_csv(OUTPUT_DIR / "feature_importances.csv")

def predict_and_save(rfc: RandomForestClassifier, X_test: pd.DataFrame, test_ids: pd.Series) -> None:
    predictions = rfc.predict(X_test)
    label_names = pd.Series(predictions).map(LABEL_NAMES)

    output = pd.DataFrame({"id": test_ids, "status_group": label_names})
    output.to_csv(OUTPUT_DIR / "predictions.csv", index=False)
