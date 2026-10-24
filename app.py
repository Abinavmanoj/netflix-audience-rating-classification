
import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Netflix Rating Classifier",
    page_icon="🎬",
    layout="centered"
)

@st.cache_resource
def load_model():
    return joblib.load("random_forest_baseline.joblib")

model = load_model()

st.title("🎬 Netflix Content Rating Classifier")
st.write(
    "Predict a Netflix content-rating category from "
    "basic title metadata using a Random Forest model."
)

content_type = st.selectbox(
    "Content type",
    ["Movie", "TV Show"]
)

release_year = st.number_input(
    "Release year",
    min_value=1925,
    max_value=2026,
    value=2020
)

duration = st.text_input(
    "Duration",
    placeholder="e.g. 90 min or 2 Seasons"
)

country = st.text_input(
    "Country",
    placeholder="e.g. United States"
)

genres = st.text_input(
    "Genres",
    placeholder="e.g. Dramas, Comedies"
)

if st.button("Predict rating"):
    if not duration.strip() or not country.strip() or not genres.strip():
        st.warning("Please complete all fields.")
    else:
        input_data = pd.DataFrame([{
            "type": content_type,
            "release_year": release_year,
            "duration": duration.strip(),
            "country": country.strip(),
            "listed_in": genres.strip()
        }])

        prediction = model.predict(input_data)[0]

        st.success(f"Predicted content rating: {prediction}")
        st.caption(
            "This is a machine-learning estimate, not an official rating."
        )
