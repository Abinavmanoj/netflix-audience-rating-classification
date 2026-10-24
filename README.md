# Netflix Audience Rating Classification

A machine learning project that predicts Netflix content-rating categories using metadata such as content type, release year, duration, country, and genres.

## Project Overview

This project explores Netflix metadata and trains classification models to predict the `rating` category.

**Note:** The dataset's `rating` field represents content classifications such as TV-MA, TV-14, PG-13, and R, not viewer satisfaction scores.

## Dataset

- 8,790 titles
- 10 columns
- No missing values in the inspected dataset
- Target: `rating`

## Models

- Decision Tree Classifier
- Random Forest Classifier
- Random Forest hyperparameter tuning with cross-validation

## Baseline Results

| Model | Accuracy | Macro F1 | Weighted F1 |
|---|---:|---:|---:|
| Random Forest baseline (initial) | 47.72% | 0.3320 | 0.4873 |
| Random Forest tuned | 45.85% | 0.3088 | 0.4710 |
| Smaller Random Forest for deployment | 42.21% | 0.28 | 0.45 |

Results are from a held-out test split. The smaller deployment model uses different Random Forest parameters, so its results should be considered separately from the initial baseline.

## Technologies

- Python
- Pandas
- Scikit-learn
- Matplotlib
- Seaborn
- Streamlit
- Joblib

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Launch the application:

```bash
streamlit run app.py
```

## Limitations

- Rating categories are imbalanced.
- Rare categories have very few training examples.
- Model predictions are estimates and should not replace official content ratings.
- Performance may vary for metadata that differs from the training dataset.

## Project Status

The classification models and Streamlit prediction demo have been developed. Deployment and final evaluation are in progress.