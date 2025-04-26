# predict.py

import requests

url = "http://127.0.0.1:8000/predict"
payload = {
    "question": "What is the capital of France?"
}
response = requests.post(url, json=payload)

print("Prediction:", response.json())
