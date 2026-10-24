
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold
)
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report
)
from sklearn.base import clone
import joblib

# Load dataset
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

# Use the same split settings as the baseline
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Load and reuse the existing fitted pipeline's structure
baseline = joblib.load("random_forest_baseline.joblib")

# Clone its configuration without retaining fitted state
pipeline = clone(baseline)

# Keep the search manageable
parameter_grid = {
    "classifier__n_estimators": [100, 200],
    "classifier__max_depth": [None, 20],
    "classifier__min_samples_leaf": [1, 2]
}

cv = StratifiedKFold(
    n_splits=3,
    shuffle=True,
    random_state=42
)

search = GridSearchCV(
    estimator=pipeline,
    param_grid=parameter_grid,
    scoring="f1_macro",
    cv=cv,
    n_jobs=-1,
    verbose=1
)

print("Tuning Random Forest...")
search.fit(X_train, y_train)

print("\nBest parameters:")
print(search.best_params_)

print("\nBest cross-validation macro F1:")
print(round(search.best_score_, 4))

# Evaluate once on the held-out test set
best_model = search.best_estimator_
predictions = best_model.predict(X_test)

print("\nTest accuracy:", round(
    accuracy_score(y_test, predictions), 4
))

print("Test macro F1:", round(
    f1_score(y_test, predictions, average="macro", zero_division=0), 4
))

print("Test weighted F1:", round(
    f1_score(y_test, predictions, average="weighted", zero_division=0), 4
))

print("\nClassification report:")
print(classification_report(
    y_test,
    predictions,
    zero_division=0
))

joblib.dump(best_model, "random_forest_tuned.joblib")
print("\nSaved tuned model: random_forest_tuned.joblib")
