# api/app.py

from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI app
app = FastAPI()

# Load the saved model and vectorizer
model = joblib.load("api/models/svm.pkl")
vectorizer = joblib.load("api/models/tfidf.pkl")

print("✅ Model and Vectorizer loaded!")
print("✅ Checking if TF-IDF is fitted...")

try:
    vectorizer.transform(["test"])  # Should not throw error
    print("✅ TF-IDF vectorizer is fitted and ready!")
except Exception as e:
    print("❌ TF-IDF loading error:", e)

# Define the request body with BaseModel for FastAPI UI
class QuestionRequest(BaseModel):
    question: str

# Define the prediction endpoint
@app.post("/predict")
async def predict_question(request: QuestionRequest):
    # Use request.question directly (Pydantic validated)
    question_text = request.question

    # Vectorize
    question_vec = vectorizer.transform([question_text])

    # Predict
    prediction = model.predict(question_vec)

    # Map integer labels to Bloom’s Taxonomy
    label_mapping = {
        0: "Remember",
        1: "Understand",
        2: "Apply",
        3: "Analyze",
        4: "Evaluate",
        5: "Create"
    }
    predicted_label = label_mapping.get(prediction[0], "Unknown")

    return {"prediction": predicted_label}
