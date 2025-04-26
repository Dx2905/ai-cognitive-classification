# streamlit_app.py

import streamlit as st
import requests

# FastAPI endpoint URL
API_URL = "http://127.0.0.1:8000/predict"

st.set_page_config(page_title="AI Cognitive Classification", page_icon="🧠")

st.title("🧠 AI Cognitive Classification (Bloom's Taxonomy NLP)")
st.markdown(
    """
    Enter an **educational question** below, and the model will classify it into one of Bloom's Taxonomy levels.
    """
)

# User input
user_input = st.text_area("📚 Your Question:", height=150)

# Predict button
if st.button("🔎 Predict Cognitive Level"):
    if not user_input.strip():
        st.warning("⚠️ Please enter a non-empty question.")
    else:
        try:
            # Prepare the JSON payload
            payload = {"question": user_input}

            # Send POST request
            response = requests.post(API_URL, json=payload)

            # Process the response
            if response.status_code == 200:
                prediction = response.json().get("prediction", "Unknown")
                st.success(f"✅ **Predicted Bloom's Cognitive Level:** `{prediction}`")
            else:
                st.error(f"❌ Server Error {response.status_code}: {response.text}")

        except requests.exceptions.RequestException as e:
            st.error(f"🚨 Failed to connect to the prediction server.\n\nError: {e}")
