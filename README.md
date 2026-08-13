# LedgerMind-AI

## Explainable AI-Powered Financial Risk Intelligence Platform

![Python](https://img.shields.io/badge/Python-3.11-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-FF4B4B.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)
![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-111111.svg)
![FAISS](https://img.shields.io/badge/FAISS-CPU-blueviolet.svg)
![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-ff69b4.svg)
![No-LangChain](https://img.shields.io/badge/Custom_RAG-Zero_LangChain-success.svg)
![Human-In-The-Loop](https://img.shields.io/badge/Workflow-Human_In_The_Loop-orange.svg)

**LedgerMind-AI** is an end-to-end production-style AI platform that unifies Machine Learning, PyTorch Deep Learning Autoencoder Anomaly Detection, Model Evaluation & Comparison, FinBERT Natural Language Processing, Explainable AI (SHAP), a custom RAG (Retrieval-Augmented Generation) pipeline built entirely without LangChain, What-If Counterfactual Risk Simulations, and a Human-in-the-Loop Analyst Review Workflow.

---

## 🌟 Comprehensive Platform Capabilities

1. **Financial Transaction Risk Prediction**:
   - Machine Learning algorithms: **Logistic Regression**, **Random Forest**, and **XGBoost**.
   - Imbalance handling: **SMOTE** oversampling and cost-sensitive class weighting.
   - Outputs: `{"risk_score": 0.82, "risk_level": "HIGH"}`.

2. **Model Performance Metrics & Comparison**:
   - Evaluates Accuracy, Precision, Recall, F1 Score, ROC-AUC, and Confusion Matrix across all trained algorithms.
   - Interactive comparative performance matrix, Plotly ROC curves, Confusion Matrix heatmaps, and feature importance rankings.
   - Metrics computed on the synthetic baseline dataset (`data/transactions.csv`).

3. **Dedicated Deep Anomaly Detection**:
   - PyTorch **Deep Autoencoder** computes MSE reconstruction loss on transaction feature vectors.
   - Normalizes raw MSE into a `[0.0, 1.0]` Anomaly Score and categorizes status into:
     - `NORMAL` (score < 0.40)
     - `SUSPICIOUS` (0.40 <= score < 0.70)
     - `HIGHLY ANOMALOUS` (score >= 0.70)
   - **Combined Risk Assessment**: Formulates final risk metric as a weighted synthesis (`65% Supervised Probability + 35% Autoencoder Anomaly Score`).

4. **Human-in-the-Loop Analyst Review Workflow**:
   - Transactions reaching `HIGH` risk level or `HIGHLY ANOMALOUS` status automatically trigger **`"Requires Human Review"`**.
   - The AI assists but does **NOT** make autonomous binding decisions.
   - Analyst Review Portal in Streamlit allows compliance officers to inspect full risk telemetry, SHAP factors, RAG evidence, news sentiment, AI mitigation steps, add comments, and log binding decisions: **`APPROVE`**, **`REJECT`**, or **`INVESTIGATE`**.
   - Decisions persist with UTC timestamps to `data/analyst_reviews.json`.

5. **Explainable AI (SHAP)**:
   - **SHAP (SHapley Additive exPlanations)** attributions answer *"Why is this transaction risky?"*, detailing positive risk-elevating (`+`) and negative risk-mitigating (`-`) drivers.

6. **Financial Document Intelligence & Custom RAG (Without LangChain)**:
   - PyMuPDF (`fitz`) document parsing, custom sliding-window text chunking, `SentenceTransformers` (`all-MiniLM-L6-v2`) embeddings, and **FAISS** vector store.
   - Grounded RAG answer generator outputting clear evidence citations (e.g., `Annual Report Page 12`).

7. **Financial News Sentiment (FinBERT)**:
   - **FinBERT** (`ProsusAI/finbert`) financial NLP pipeline extracting Sentiment, Corporate Entities, and Risk Impact (`High Risk`, `Medium Risk`, `Low Risk`).

8. **What-If Counterfactual Scenario Simulator**:
   - Simulates parameter shifts (interest rate hikes, transaction amount surges, income changes) to compare Before vs After risk scores.

---

## 🏗️ Integrated Architecture Diagram

```
Financial Data / Transaction
        │
        ▼
Data Preprocessing & Feature Engineering
        │
        ▼
┌───────────────────────────────┬───────────────────────────────┐
│ Supervised Risk Prediction    │ Unsupervised Deep Autoencoder │
│ (LogisticReg / RF / XGBoost)  │ Anomaly Detection (PyTorch)   │
└───────────────┬───────────────┴───────────────┬───────────────┘
                │                               │
                └───────────────┬───────────────┘
                                ▼
                    Combined Risk Assessment
                                │
                                ▼
                    SHAP Risk Factor Attribution
                                │
        ┌───────────────────────┴───────────────────────┐
        ▼                                               ▼
Financial Documents → RAG Evidence    Financial News → FinBERT Sentiment
        │                                               │
        └───────────────────────┬───────────────────────┘
                                ▼
                   What-If Scenario Simulation
                                │
                                ▼
              AI Risk Mitigation Recommendations
                                │
                                ▼
            Human Analyst Review (APPROVE/REJECT/INVESTIGATE)
                                │
                                ▼
                     Final Risk Dashboard
```

---

## 📂 Project Directory Structure

```
LedgerMind-AI/
├── README.md
├── requirements.txt
├── .gitignore
├── data/
│   ├── transactions.csv
│   ├── financial_news.csv
│   ├── analyst_reviews.json
│   └── documents/
│       └── annual_risk_report_2025.txt
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Model_Training.ipynb
│   └── 03_NLP_Analysis.ipynb
├── src/
│   ├── data_processing/
│   │   ├── preprocess.py
│   │   ├── feature_engineering.py
│   │   └── generate_sample_data.py
│   ├── models/
│   │   ├── autoencoder.py
│   │   ├── risk_model.py
│   │   ├── train.py
│   │   ├── predict.py
│   │   └── human_review.py
│   ├── nlp/
│   │   ├── document_parser.py
│   │   ├── embeddings.py
│   │   └── news_sentiment.py
│   ├── rag/
│   │   ├── chunking.py
│   │   ├── vector_store.py
│   │   └── retrieval.py
│   ├── explainability/
│   │   └── shap_explainer.py
│   ├── simulation/
│   │   └── what_if.py
│   └── recommendation/
│       └── advisor.py
├── api/
│   └── main.py
├── frontend/
│   └── app.py
├── models/
│   └── risk_model.pkl
├── tests/
│   ├── test_data_processing.py
│   ├── test_models.py
│   ├── test_model_eval.py
│   ├── test_anomaly.py
│   ├── test_human_review.py
│   ├── test_rag.py
│   ├── test_nlp.py
│   └── test_api.py
└── docker/
    ├── Dockerfile
    └── docker-compose.yml
```

---

## 🚀 Quickstart Guide

### 1. Setup Environment

```bash
git clone https://github.com/your-username/LedgerMind-AI.git
cd LedgerMind-AI

python -m venv .venv
# Windows:
.venv\Scripts\activate
# Linux/macOS:
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Generate Data & Train Models

```bash
python src/data_processing/generate_sample_data.py
python src/models/train.py
```

### 3. Run FastAPI Backend

```bash
uvicorn api.main:app --reload --port 8000
```
API Documentation: [http://localhost:8000/docs](http://localhost:8000/docs)

### 4. Run Streamlit Dashboard

```bash
streamlit run frontend/app.py
```
Dashboard: [http://localhost:8501](http://localhost:8501)

### 5. Run Complete Test Suite

```bash
pytest tests/ -v
```

---

## 📌 API Endpoints

| Method | Endpoint | Description |
| :--- | :--- | :--- |
| `POST` | `/predict-risk` | Returns risk score, anomaly score, combined score, & human review flag |
| `POST` | `/explain-risk` | Generates SHAP-based feature importance attributions |
| `GET` | `/model-evaluation` | Returns Accuracy, Precision, Recall, F1, ROC-AUC, & Confusion Matrix for all models |
| `POST` | `/retrieve-evidence`| Custom RAG search & evidence citation answer generation |
| `POST` | `/analyze-news` | FinBERT financial news sentiment & risk impact analysis |
| `POST` | `/simulate` | What-if counterfactual scenario simulator |
| `POST` | `/recommend` | AI recommendation engine for risk mitigation |
| `POST` | `/review-transaction`| Records analyst review decision (`APPROVE`, `REJECT`, `INVESTIGATE`) |
| `GET` | `/reviews` | Retrieves log of all recorded analyst reviews |
| `GET` | `/health` | Server health check endpoint |

---

## 📄 License
Licensed under the [MIT License](LICENSE).
