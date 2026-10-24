
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay
)
import matplotlib.pyplot as plt

# 1. Load dataset
df = pd.read_csv("Dataset.csv")

# 2. Select features and target
features = [
    "type",
    "release_year",
    "duration",
    "country",
    "listed_in"
]
target = "rating"

X = df[features].copy()
y = df[target].copy()

# Convert numeric feature to numeric values
X["release_year"] = pd.to_numeric(
    X["release_year"], errors="coerce"
)

# 3. Split data while preserving class proportions
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Preprocess features
categorical_features = ["type", "duration", "country"]
text_feature = "listed_in"

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ]),
            categorical_features
        ),
        (
            "genres",
            TfidfVectorizer(),
            text_feature
        ),
        (
            "year",
            SimpleImputer(strategy="median"),
            ["release_year"]
        )
    ]
)

# 5. Define models
models = {
    "Decision Tree": DecisionTreeClassifier(
        random_state=42,
        class_weight="balanced"
    ),
    "Random Forest": RandomForestClassifier(
        n_estimators=100,
        max_depth=20,
min_samples_leaf=2,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )
}

# 6. Train and evaluate
results = {}

for name, model in models.items():
    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", model)
    ])

    print(f"\nTraining {name}...")
    pipeline.fit(X_train, y_train)
    if name == "Random Forest":
       import joblib
       joblib.dump(pipeline, "random_forest_baseline.joblib")
    print("Random Forest model saved successfully!")
    predictions = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)
    results[name] = accuracy

    print(f"\n{name} accuracy: {accuracy:.4f}")
    print(classification_report(
        y_test,
        predictions,
        zero_division=0
    ))

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        xticks_rotation="vertical",
        include_values=False
    )
    plt.title(f"{name} Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        name.lower().replace(" ", "_") + "_confusion_matrix.png",
        dpi=150
    )
    plt.close()

# 7. Compare accuracy
print("\nModel comparison:")
for name, accuracy in results.items():
    print(f"{name}: {accuracy:.4f}")

import joblib

# Save the trained Random Forest pipeline
# Add this inside the model loop after fitting,
# or fit and save the Random Forest separately.
