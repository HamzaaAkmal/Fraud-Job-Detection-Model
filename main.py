import streamlit as st
import numpy as np
import pickle
from scipy.sparse import hstack


with open("svm_fraud_model.pkl", "rb") as f:
    model, vectorizer = pickle.load(f)

st.set_page_config(page_title="🕵️‍♂️ Fraud Job Detector", layout="centered")
st.title("🚩 Job Posting Fraud Detector")
st.markdown("Enter job details to detect if it's **fraudulent** or **legitimate**.")

with st.form("job_form"):
    st.subheader("📋 Job Information")
    
    title = st.text_input("Job Title", "Mystery Shopper - Easy Money")
    description = st.text_area("Job Description", "We send you checks; you cash them, buy products, and report back. Get $300 per task.")
    requirements = st.text_area("Requirements", "Must be 18+ Willing to handle cash")
    benefits = st.text_area("Benefits", "Fast payment")
    company_profile = st.text_area("Company Profile", "MysteryShopOnline - 100% guaranteed payments.")
    
    st.subheader("🔢 Additional Fields")
    telecommuting = st.selectbox("Is it a remote job?", [0, 1])
    has_company_logo = st.selectbox("Company has a logo?", [0, 1])
    has_questions = st.selectbox("Has screening questions?", [0, 1])
    
    submitted = st.form_submit_button("Predict")

if submitted:
    new_text = title + " " + description + " " + requirements + " " + benefits + " " + company_profile
    
    X_text = vectorizer.transform([new_text])
 
    X_numeric = np.array([[telecommuting, has_company_logo, has_questions]])

    X_new = hstack([X_text, X_numeric])
    
  
    prediction = model.predict(X_new)
    
    if prediction == 1:
        st.error("⚠️ This job posting is likely **fraudulent**.")
    else:
        st.success("✅ This job posting appears **legitimate**.")
