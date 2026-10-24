
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score
)

# Load the dataset
df = pd.read_csv("Dataset.csv")

features = [
    "type",
    "release_year",
    "duration",
    "country",
    "listed_in"
]

X = df[features].copy()
y = df["rating"].copy()

X["release_year"] = pd.to_numeric(
    X["release_year"], errors="coerce"
)

# Reproduce the original test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

models = {
    "Random Forest Baseline": joblib.load(
        "random_forest_baseline.joblib"
    ),
    "Random Forest Tuned": joblib.load(
        "random_forest_tuned.joblib"
    )
}

rows = []

for name, model in models.items():
    predictions = model.predict(X_test)

    rows.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Macro Precision": precision_score(
            y_test, predictions, average="macro",
            zero_division=0
        ),
        "Macro Recall": recall_score(
            y_test, predictions, average="macro",
            zero_division=0
        ),
        "Macro F1": f1_score(
            y_test, predictions, average="macro",
            zero_division=0
        ),
        "Weighted F1": f1_score(
            y_test, predictions, average="weighted",
            zero_division=0
        )
    })

results = pd.DataFrame(rows)
print(results.round(4).to_string(index=False))

results.to_csv("model_comparison.csv", index=False)
print("\nSaved: model_comparison.csv")
