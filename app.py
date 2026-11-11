import joblib
import pandas as pd
import streamlit as st

# ==========================
# Load the trained model and feature order
# ==========================
@st.cache_resource
def load_model():
    model = joblib.load("personality_model.pkl")
    with open("final_feature_list.txt", "r") as f:
        feature_columns = [line.strip() for line in f if line.strip()]
    return model, feature_columns


model, feature_columns = load_model()

YES_NO = {"No": 0, "Yes": 1}
LABELS = {0: "Extrovert", 1: "Introvert"}

# ==========================
# Page layout
# ==========================
st.set_page_config(page_title="Personality Prediction", layout="centered")
st.title("Personality Type Prediction")
st.markdown("Answer a few questions about social habits to predict whether a person is an **introvert** or an **extrovert**.")

with st.form("personality_form"):
    time_alone = st.slider("Hours spent alone per day", 0, 11, 2)
    stage_fear = st.radio("Has stage fear?", list(YES_NO), horizontal=True)
    social_events = st.slider("Social event attendance (0 = never, 10 = very often)", 0, 10, 5)
    going_outside = st.slider("How often they go outside (0 = rarely, 7 = daily)", 0, 7, 4)
    drained = st.radio("Feels drained after socialising?", list(YES_NO), horizontal=True)
    friends = st.slider("Size of close friends circle", 0, 15, 8)
    posts = st.slider("Social media post frequency (0 = never, 10 = very often)", 0, 10, 5)
    submitted = st.form_submit_button("Predict personality")

# ==========================
# Prediction
# ==========================
if submitted:
    inputs = {
        "Time_spent_Alone": time_alone,
        "Stage_fear": YES_NO[stage_fear],
        "Social_event_attendance": social_events,
        "Going_outside": going_outside,
        "Drained_after_socializing": YES_NO[drained],
        "Friends_circle_size": friends,
        "Post_frequency": posts,
    }
    # Keep the exact column order the model was trained on
    input_df = pd.DataFrame([inputs])[feature_columns]

    prediction = int(model.predict(input_df)[0])
    probabilities = model.predict_proba(input_df)[0]

    st.success(f"Predicted personality: **{LABELS[prediction]}**")

    st.markdown("### Prediction probabilities")
    prob_df = pd.DataFrame(
        {"Personality": [LABELS[int(c)] for c in model.classes_], "Probability": probabilities}
    )
    st.bar_chart(prob_df.set_index("Personality"))
