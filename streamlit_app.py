# streamlit_app.py

import streamlit as st
import requests

# FastAPI endpoint URL
API_URL = "http://127.0.0.1:8000/predict"

st.title("🧠 AI Cognitive Classification (Bloom's Taxonomy NLP)")
st.write("Enter an educational question below. The model will predict its Bloom's Taxonomy cognitive level!")

# User input
user_input = st.text_area("Enter your question here:")

if st.button("Predict"):
    if user_input.strip() == "":
        st.warning("⚠️ Please enter a valid question.")
    else:
        # Prepare request body
        payload = {"question": user_input}

        try:
            # Send request to FastAPI backend
            response = requests.post(API_URL, json=payload)
            
            if response.status_code == 200:
                prediction = response.json()["prediction"]
                st.success(f"✅ Predicted Cognitive Level: **{prediction}**")
            else:
                st.error(f"❌ Server Error: {response.status_code}")

        except Exception as e:
            st.error(f"❌ Error connecting to API: {e}")
