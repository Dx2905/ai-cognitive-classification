# 👤 AI Cognitive Classification (Bloom’s Taxonomy NLP)

## 📌 Project Overview
This project builds a **machine learning-based cognitive classification system** that categorizes educational questions according to **Bloom’s Taxonomy**. The solution uses **TF-IDF vectorization**, **Word2Vec embeddings**, and **supervised learning models** like **SVM** and **Random Forest**. It incorporates **model explainability (SHAP)**, **real-time serving (FastAPI)**, **streamlit-based demo interface**, **ML experiment tracking (Weights & Biases)**, and **data/model version control (DVC)**.

## 📄 Table of Contents
- [Project Overview](#project-overview)
- [Key Features](#key-features)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Dataset](#dataset)
- [Results & Performance](#results--performance)
- [Future Work](#future-work)
- [License](#license)

---

## 🎯 Key Features
- ✨ **Educational Question Classification** into Bloom’s cognitive levels.
- 📊 **Trained on TF-IDF & Word2Vec** features.
- ✅ **Models Used**: SVM (RBF kernel), Random Forest.
- 🔄 **Weekly retraining & dataset ingestion automated** via Airflow.
- 🎨 **SHAP Explainability** for model decisions.
- 🔗 **FastAPI API** for real-time inference.
- 👨‍💨 **Streamlit Demo** frontend for user interaction.
- 📊 **Weights & Biases Integration** for experiment tracking.
- 🌀 **Data/Model Versioning with DVC**.

---

## 💡 Technologies Used
- **Programming**: Python
- **Libraries**: scikit-learn, pandas, numpy, matplotlib, seaborn
- **ML Serving**: FastAPI, Uvicorn
- **UI**: Streamlit
- **Tracking**: Weights & Biases
- **Versioning**: DVC
- **Explainability**: SHAP
- **Orchestration**: Airflow (for retraining automation)

---

## 🛠️ Installation

1. **Clone the Repository**:
```bash
git clone https://github.com/Dx2905/ai-cognitive-classification.git
cd ai-cognitive-classification
```

2. **Create and Activate Virtual Environment**:
```bash
python -m venv venv
source venv/bin/activate  # For Windows: venv\Scripts\activate
```

3. **Install Dependencies**:
```bash
pip install -r requirements.txt
```

4. **(Optional) Set up DVC**:
```bash
dvc pull
```

5. **(Optional) Set up Airflow** (for retraining DAG):
Follow airflow docs to initialize airflow and place DAG under `airflow/dags/`.

---

## 🚀 Usage

### 1. **Train Model**
```bash
python src/train.py --model svm
```
- Saves model and TF-IDF vectorizer in `api/models/`.
- Tracks experiments on Weights & Biases (wandb).

### 2. **Run FastAPI Backend**
```bash
python -m uvicorn api.app:app --reload
```
- Starts local API server on `http://127.0.0.1:8000`.

### 3. **Launch Streamlit Frontend**
```bash
streamlit run streamlit_app.py
```
- Interactively predict Bloom’s level from your own question input.

---

## 📈 Dataset
- **Source**: Collected academic databases + public question banks + augmented synthetic questions.
- **Classes**:
  - Remember
  - Understand
  - Apply
  - Analyze
  - Evaluate
  - Create

---

## 📊 Results & Performance
- **SVM (RBF Kernel)** achieved ~**99.69% accuracy**.
- **Evaluation Metrics**:
  - ROC-AUC
  - Cohen’s Kappa
  - Precision, Recall, F1-score

| Model               | Accuracy |
|---------------------|----------|
| SVM (RBF Kernel)     | 99.69%   |
| Random Forest        | 97.20%   |


### SHAP Explainability
- Computed SHAP values on validation samples to understand word contributions toward predictions.
- Local and Global explanations available.

---

## 🔮 Future Improvements
- Integrate Word2Vec feature augmentation into FastAPI serving.
- Full retraining pipelines via Airflow.
- Deploy complete system (FastAPI + Streamlit) using Docker.
- Integrate feedback loop into active learning workflow.

---

## 📅 License
This project is licensed under the **MIT License**.

---

✨ **If you found this project useful, give it a star on GitHub!** ✨


