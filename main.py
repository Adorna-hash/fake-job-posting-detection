# streamlit_app.py

import streamlit as st  # 🔥 MUST BE FIRST
st.set_page_config(page_title="Fake Job Detector", layout="wide")  # ✅ MUST come immediately after import

import pickle
import numpy as np
import pandas as pd
from scipy.sparse import hstack

# --------------------
# Load Saved Artifacts
# --------------------
@st.cache_resource
def load_model():
    with open('fraud_model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('encoder.pkl', 'rb') as f:
        encoder = pickle.load(f)
    with open('tfidf.pkl', 'rb') as f:
        tfidf = pickle.load(f)
    return model, encoder, tfidf

model, encoder, tfidf = load_model()

# --------------------
# App Title
# --------------------
st.title("🕵️ Fake Job Posting Detector")


# Input Form
with st.form("job_form"):
    title = st.text_input("Job Title")
    description = st.text_area("Job Description")
    requirements = st.text_area("Job Requirements")

    col1, col2, col3 = st.columns(3)

    with col1:
        employment_type = st.selectbox("Employment Type", ['Unknown', 'Full-time', 'Part-time', 'Contract', 'Temporary', 'Other'])
        telecommuting = st.selectbox("Telecommuting", [0, 1])
    with col2:
        required_experience = st.selectbox("Required Experience", ['Unknown', 'Internship', 'Entry level', 'Associate', 'Mid-Senior level', 'Director', 'Executive'])
        has_company_logo = st.selectbox("Has Company Logo", [0, 1])
    with col3:
        required_education = st.selectbox("Required Education", ['Unknown', 'High School', 'Some College', 'Associate Degree', 'Bachelor’s Degree', 'Master’s Degree', 'Doctorate'])
        has_questions = st.selectbox("Has Screening Questions", [0, 1])

    industry = st.text_input("Industry (optional)", value="Unknown")
    function = st.text_input("Function (optional)", value="Unknown")

    submit = st.form_submit_button("Predict")

# --------------------
# Prediction Logic
# --------------------
if submit:
    # Structured features
    cat_features = pd.DataFrame([{
        'employment_type': employment_type,
        'required_experience': required_experience,
        'required_education': required_education,
        'industry': industry,
        'function': function
    }])

    bin_features = pd.DataFrame([{
        'telecommuting': telecommuting,
        'has_company_logo': has_company_logo,
        'has_questions': has_questions
    }])

    encoded_cat = encoder.transform(cat_features)
    structured_input = pd.concat([bin_features.reset_index(drop=True),
                                  pd.DataFrame(encoded_cat,
                                               columns=encoder.get_feature_names_out(),
                                               index=bin_features.index)],
                                 axis=1)

    # Text features
    full_text = title + " " + description + " " + requirements
    text_input = tfidf.transform([full_text])

    # Combine
    final_input = hstack([structured_input.to_numpy(), text_input])

    # Predict
    pred = model.predict(final_input)[0]
    prob = model.predict_proba(final_input)[0][1]
# -------------------------------
# Prediction & Display with Threshold Logic
# -------------------------------
# -------------------------------
# Prediction & Display with Confidence Threshold Logic
# -------------------------------
threshold = 0.6
fraud_prob = model.predict_proba(final_input)[0][1]

if fraud_prob >= threshold:
    label = "Fraudulent"
    confidence = fraud_prob
else:
    label = "Genuine"
    confidence = 1 - fraud_prob

# Final decision logic
if confidence >= threshold:
    if label == "Fraudulent":
        st.error(f"🔍 Prediction: {label}\n\n⚠️ This job posting is likely fraudulent.\n\n**Confidence:** {confidence:.2%}")
    else:
        st.success(f"🔍 Prediction: {label}\n\n✅ This job posting appears genuine.\n\n**Confidence:** {confidence:.2%}")
else:
    # Flip label if confidence is too low, don't show confidence
    flipped_label = "Genuine" if label == "Fraudulent" else "Fraudulent"
    if flipped_label == "Fraudulent":
        st.error(f"🔍 Prediction: {flipped_label}\n\n⚠️ This job posting is likely fraudulent.")
    else:
        st.success(f"🔍 Prediction: {flipped_label}\n\n✅ This job posting appears genuine.")

# Always show confidence bar
st.progress(int(confidence * 100))





