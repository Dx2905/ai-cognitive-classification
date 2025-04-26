# src/train.py

import os
import pandas as pd
import numpy as np
import argparse
import joblib
import warnings
import shap
import wandb

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, cohen_kappa_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# Argument parser
parser = argparse.ArgumentParser()
parser.add_argument('--data', type=str, default='data/Question.xlsx', help='Path to dataset')
parser.add_argument('--model', type=str, default='svm', choices=['svm', 'random_forest'], help='Model type')
parser.add_argument('--save_path', type=str, default='api/models/', help='Directory to save model and vectorizer')
args = parser.parse_args()

# Load and preprocess data
df = pd.read_excel(args.data)
df.dropna(inplace=True)

print("✅ Columns:", df.columns)

# Text and label encoding
X = df['Questions']
y = LabelEncoder().fit_transform(df['Label'])

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y, test_size=0.2, random_state=42)

# TF-IDF Vectorizer
tfidf = TfidfVectorizer(max_features=3000)
tfidf.fit(X_train)  # ✅ Ensure fitted
print("✅ TF-IDF Vectorizer fitted.")

# Transform
X_train_vec = tfidf.transform(X_train)
X_test_vec = tfidf.transform(X_test)

# Model selection
if args.model == 'svm':
    model = SVC(kernel='rbf', probability=True, random_state=42)
elif args.model == 'random_forest':
    model = RandomForestClassifier(n_estimators=100, random_state=42)

# Model training
model.fit(X_train_vec, y_train)
print("✅ Model trained.")

# Evaluation
y_pred = model.predict(X_test_vec)
y_proba = model.predict_proba(X_test_vec)

print("\nClassification Report:\n", classification_report(y_test, y_pred))
print("ROC-AUC:", roc_auc_score(y_test, y_proba, multi_class='ovr'))
print("Cohen's Kappa:", cohen_kappa_score(y_test, y_pred))

# (Optional) SHAP Explainability
# print("Computing SHAP values... (this is commented out)")
# explainer = shap.KernelExplainer(model.predict_proba, X_train_vec[:100])
# shap_values = explainer.shap_values(X_test_vec[:50])
# shap.summary_plot(shap_values, X_test_vec[:50], feature_names=tfidf.get_feature_names_out(), show=False)

# Save ONLY after fitting and training is done
os.makedirs(args.save_path, exist_ok=True)
joblib.dump(model, os.path.join(args.save_path, f"{args.model}.pkl"))
joblib.dump(tfidf, os.path.join(args.save_path, "tfidf.pkl"))
print(f"✅ Model and TF-IDF saved to {args.save_path}")

# (Optional) W&B Logging
if os.getenv("WANDB_API_KEY"):
    wandb.init(project="ai-cognitive-bloom", config=args)
    wandb.sklearn.plot_classifier(model, X_train_vec, X_test_vec, y_train, y_test, y_pred,y_proba, labels=np.unique(y))
    wandb.finish()



