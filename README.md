# 🛡️ LedgerMind AI — Financial Risk & Compliance Intelligence Platform

[![Python](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.95+-009688.svg)](https://fastapi.tiangolo.com/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.24+-FF4B4B.svg)](https://streamlit.io/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-EE4C2C.svg)](https://pytorch.org/)
[![XGBoost](https://img.shields.io/badge/XGBoost-1.7+-111111.svg)](https://xgboost.readthedocs.io/)
[![FAISS](https://img.shields.io/badge/FAISS-CPU-blueviolet.svg)](https://github.com/facebookresearch/faiss)
[![SHAP](https://img.shields.io/badge/SHAP-Explainable_AI-ff69b4.svg)](https://shap.readthedocs.io/)
[![Custom RAG](https://img.shields.io/badge/Custom_RAG-Zero_LangChain-success.svg)](https://github.com/shashicharan203/RiskLens-AI)
[![Human-In-The-Loop](https://img.shields.io/badge/Workflow-Human_In_The_Loop-orange.svg)](https://github.com/shashicharan203/RiskLens-AI)

**LedgerMind AI** is an enterprise-grade financial risk intelligence platform designed to replace black-box risk scoring with **Explainable AI (XAI)** and **Policy-Grounded Retrieval-Augmented Generation (RAG)**. 

The platform integrates supervised machine learning, PyTorch deep autoencoder behavior anomaly detection, SHAP-based feature attributions with business-friendly labels, custom FAISS-backed RAG (built without LangChain), FinBERT financial news sentiment analysis, What-If counterfactual scenario simulations, and a Human-in-the-Loop compliance review workflow.

---

## ⚡ Executive Summary (30-Second Overview)

In traditional banking systems, machine learning models output a numerical risk score (e.g. `82% Risk`), leaving compliance officers without clear context or regulatory justification.

**LedgerMind AI solves this using a 4-tier decision support architecture:**

1. **Supervised ML & Anomaly Detection**: Predicts customer risk score (XGBoost/Random Forest) and behavioral anomaly score (PyTorch Deep Autoencoder).
2. **SHAP Explainability**: Translates complex model features into clear, business-friendly risk drivers (e.g., *"Multiple Failed Login Attempts"*, *"High Device Risk"*).
3. **Policy-Grounded RAG Evidence**: Searches uploaded bank policy documents for compliance guidelines matching that specific customer's actual risk profile (e.g., *"Why is CUST_0005 considered high risk according to the uploaded policy?"*).
4. **Human Analyst Review**: Flags high-risk cases for analyst approval, rejection, or escalation (`APPROVE`, `REJECT`, `ESCALATE`).

> ℹ️ **Core Product Rule:** ML predicts customer risk. SHAP explains why the model predicted that risk. RAG does **NOT** generate or calculate risk scores—RAG searches bank/company policies to retrieve supporting compliance evidence grounded in that customer's actual risk factors.

---

## 🏗️ End-to-End Core Workflow

```
                  ┌────────────────────────────────────────┐
                  │ Transaction Data (CSV / Custom Entry)  │
                  └───────────────────┬────────────────────┘
                                      │
                                      ▼
                        Supervised ML Risk Prediction
                        (XGBoost / Random Forest)
                                      │
                                      ▼
                          Risk Score & Risk Level
                                      │
                                      ▼
                        PyTorch Deep Autoencoder
                        Anomaly Detection
                                      │
                                      ▼
                           SHAP Explainability
                     (Business-Friendly Risk Drivers)
                                      │
                                      ▼
                         Customer Risk Context
            (Customer ID, Risk Score, Anomaly Status, SHAP Factors)
                                      │
                                      ▼
               ┌──────────────────────┴──────────────────────┐
               │                                             │
               ▼                                             ▼
   Bank / Company Policy (PDF/TXT)               Financial News Articles
               │                                             │
               ▼                                             ▼
   FAISS Vector Store Retrieval                  FinBERT Market Sentiment
        (Zero-LangChain)                                     │
               │                                             │
               ▼                                             │
   Policy Evidence & Citations                               │
  (Answer, Excerpt, Source, Page)                            │
               │                                             │
               └──────────────────────┬──────────────────────┘
                                      │
                                      ▼
                       What-If Scenario Simulation
                       (Counterfactual Deltas)
                                      │
                                      ▼
                          Human Analyst Review
                     (APPROVE / REJECT / ESCALATE)
```

---

## 🌟 Key Platform Features

### 1. Multi-Model Risk Prediction & Comparative Benchmarking
- Implements **Logistic Regression**, **Random Forest**, and **XGBoost** with SMOTE oversampling to handle severe transaction imbalance.
- API returns comprehensive model evaluation metrics (`Accuracy`, `Precision`, `Recall`, `F1-Score`, `ROC-AUC`, and ROC curves) comparing performance across all algorithms.

### 2. PyTorch Deep Autoencoder Anomaly Detection
- Unsupervised PyTorch Deep Autoencoder trained to reconstruct normal financial behavior patterns.
- MSE reconstruction loss is normalized into a `[0.0, 1.0]` Anomaly Score:
  - `NORMAL` (Score < 0.40)
  - `SUSPICIOUS` (0.40 ≤ Score < 0.70)
  - `HIGHLY ANOMALOUS` (Score ≥ 0.70)
- **Combined Risk Metric**: Blends supervised prediction probability (65%) with deep reconstruction anomaly score (35%).

### 3. Business-Friendly SHAP Explainability
- Calculates exact feature contribution values (Shapley values) for individual transactions.
- Automatically maps technical column names to readable business terms in text lists and interactive Plotly contribution charts:
  - `failed_login_attempts` ➔ `Multiple Failed Login Attempts`
  - `device_risk_score` ➔ `High Device Risk`
  - `credit_utilization` ➔ `High Credit Utilization`
  - `debt_to_income` ➔ `High Debt-to-Income Ratio`
  - `transaction_amount` ➔ `Large Transaction Amount`
  - `account_age_days` ➔ `Account Age`

### 4. Grounded Customer-Specific RAG Policy Search (No LangChain)
- Built entirely without LangChain or LlamaIndex using PyMuPDF (`fitz`), custom sliding-window text chunking, `SentenceTransformers` (`all-MiniLM-L6-v2`), and **FAISS** vector indexing.
- Constructs dynamic context from the currently selected customer's actual telemetry:
  ```json
  {
    "customer_id": "CUST_0005",
    "risk_score": 0.82,
    "risk_level": "HIGH",
    "anomaly_status": "HIGHLY ANOMALOUS",
    "important_factors": ["Multiple Failed Login Attempts", "High Device Risk"]
  }
  ```
- Executes queries such as *"Why is CUST_0005 considered high risk according to the uploaded policy?"* and returns grounded policy explanations with source document and page/section citations.

### 5. FinBERT Financial News Sentiment Analysis
- Uses **FinBERT** (`ProsusAI/finbert`) to analyze market news and regulatory announcements.
- Outputs sentiment classification (`Negative`, `Neutral`, `Positive`), target corporate entity, and overall market risk impact level.

### 6. What-If Counterfactual Scenario Simulator
- Allows risk officers to adjust financial parameters (Surge in transaction amount, income changes, interest rate hikes, credit utilization spikes) to simulate counterfactual risk score deltas before executing transactions.

### 7. Human-in-the-Loop Analyst Portal
- Automatically flags high-risk or highly anomalous transactions with a **`REQUIRES HUMAN REVIEW`** banner.
- Analysts review customer case details, inspect SHAP factors and policy evidence cards, submit audit notes, and log binding decisions: **`APPROVE`**, **`REJECT`**, or **`ESCALATE`**.
- Decision history persists to `data/analyst_reviews.json` with UTC timestamps.

---

## 🛠️ Technology Stack

| Layer | Technologies Used |
| :--- | :--- |
| **Backend API** | Python 3.11, FastAPI, Uvicorn, Pydantic |
| **Frontend Dashboard** | Streamlit, Plotly Express & Graph Objects, HTML5/CSS3 |
| **Supervised Machine Learning** | XGBoost, Scikit-Learn, Imbalanced-Learn (SMOTE) |
| **Deep Learning & Anomaly** | PyTorch, PyTorch Neural Networks (`nn.Module`) |
| **Explainable AI (XAI)** | SHAP (SHapley Additive exPlanations) |
| **NLP & FinBERT** | HuggingFace Transformers, PyTorch, FinBERT |
| **RAG & Embeddings** | FAISS (Facebook AI Similarity Search), SentenceTransformers, PyMuPDF (`fitz`) |
| **Testing** | Pytest, FastAPI TestClient |

---

## 🔌 API Endpoints Reference

All backend endpoints are hosted on FastAPI at `http://127.0.0.1:8001`:

| Method | Endpoint | Request Payload / Description |
| :--- | :--- | :--- |
| `GET` | `/health` | Server health check (`{"status": "healthy"}`) |
| `GET` | `/` | Platform meta info & Swagger links |
| `POST` | `/predict-risk` | Predicts risk score, anomaly score, & human review flag for a single transaction |
| `POST` | `/explain-risk` | Returns SHAP feature contribution values and important risk factors |
| `POST` | `/upload-transactions-csv` | Multipart CSV batch upload; scores every customer in the file |
| `POST` | `/upload-document` | Parses PDF/TXT policy documents, embeds text, and indexes into FAISS store |
| `POST` | `/retrieve-evidence` | Custom RAG search returning grounded policy synthesis & citation excerpts |
| `POST` | `/analyze-news` | FinBERT market news sentiment & corporate risk impact analysis |
| `POST` | `/simulate` | What-If counterfactual scenario simulator |
| `POST` | `/review-transaction` | Logs analyst compliance decision (`APPROVE`, `REJECT`, `ESCALATE`) |
| `GET` | `/reviews` | Fetches historical decision logs |
| `GET` | `/model-evaluation` | Returns comparative Accuracy, Precision, Recall, F1, & ROC curves |

---

## 📁 Repository Directory Structure

```text
LedgerMind-AI/
├── README.md                           # Project Documentation
├── requirements.txt                    # Production Dependencies
├── .gitignore                          # Environment & Cache Exclusions
├── api/
│   └── main.py                         # FastAPI Backend Application & Lazy Loaders
├── frontend/
│   └── app.py                          # Streamlit Enterprise Dashboard & Session State
├── src/
│   ├── data_processing/
│   │   ├── preprocess.py               # Feature Preprocessing & Scaling
│   │   ├── feature_engineering.py      # Ratio & Frequency Transformations
│   │   ├── input_handler.py            # CSV & PDF Upload Validation
│   │   └── generate_sample_data.py     # Synthetic Dataset Generator
│   ├── models/
│   │   ├── autoencoder.py              # PyTorch Deep Autoencoder Architecture
│   │   ├── risk_model.py               # XGBoost/Random Forest/Logistic Training
│   │   ├── train.py                    # Model Training Pipeline
│   │   ├── predict.py                  # Batch & Single Scoring Engine
│   │   └── human_review.py             # Review Store Persistence Manager
│   ├── explainability/
│   │   └── shap_explainer.py           # SHAP Explainability Engine
│   ├── nlp/
│   │   ├── document_parser.py          # PyMuPDF / Plaintext Document Reader
│   │   ├── embeddings.py               # SentenceTransformers Generator
│   │   └── news_sentiment.py           # FinBERT Sentiment Pipeline
│   ├── rag/
│   │   ├── chunking.py                 # Custom Text Chunker with Overlap
│   │   ├── vector_store.py             # FAISS Vector Index Wrapper
│   │   └── retrieval.py                # End-to-End RAG Retriever
│   ├── llm/
│   │   └── generator.py                # Grounded Response Synthesizer
│   └── simulation/
│       └── what_if.py                  # Counterfactual Scenario Simulator
├── data/
│   ├── transactions.csv                # Baseline Transaction Dataset
│   ├── financial_news.csv              # Baseline News Dataset
│   ├── analyst_reviews.json            # Human Review History Store
│   └── documents/                      # Sample Policy Documents
├── models/
│   ├── risk_model.pkl                  # Trained Machine Learning Model
│   └── vector_store/                   # Persisted FAISS Vector Index
├── tests/
│   ├── test_anomaly.py                 # PyTorch Autoencoder Tests
│   ├── test_api.py                     # FastAPI Endpoints Integration Tests
│   ├── test_data_processing.py         # Data Pipeline Tests
│   ├── test_human_review.py            # Review Store Tests
│   ├── test_input_handler.py           # File Processing Tests
│   ├── test_model_eval.py              # Evaluation Metrics Tests
│   ├── test_models.py                  # Supervised ML Tests
│   ├── test_nlp.py                     # FinBERT & Simulation Tests
│   └── test_rag.py                     # FAISS & RAG Generator Tests
└── notebooks/
    ├── 01_EDA.ipynb                    # Exploratory Data Analysis
    ├── 02_Model_Training.ipynb         # Model Training Workbench
    └── 03_NLP_Analysis.ipynb           # NLP & RAG Experiments
```

---

## 💻 Local Setup & Execution Guide

### 1. Environment Setup

Clone the repository and initialize the Python 3.11 virtual environment:

```powershell
# Clone Repository
git clone https://github.com/shashicharan203/RiskLens-AI.git
cd RiskLens-AI

# Create & Activate Virtual Environment
python -m venv .venv
.\.venv\Scripts\activate

# Install Dependencies
pip install -r requirements.txt
```

### 2. Launch FastAPI Backend Server

Run the FastAPI backend on port `8001`:

```powershell
.venv\Scripts\python.exe -m uvicorn api.main:app --reload --port 8001
```
- **Backend API**: `http://127.0.0.1:8001`
- **Interactive Swagger Docs**: `http://127.0.0.1:8001/docs`

### 3. Launch Streamlit Frontend Dashboard

In a second terminal, launch the Streamlit interface on port `8501`:

```powershell
.venv\Scripts\python.exe -m streamlit run frontend/app.py --server.port 8501
```
- **Dashboard URL**: `http://localhost:8501`

---

## 🧪 Testing Suite

Run the full automated test suite containing 25 unit and API integration tests:

```powershell
.venv\Scripts\python.exe -m pytest tests/ -v
```

> **Test Suite Output**: `25 passed in 47.39s (100% pass rate)`

---

## 💡 Why This Project Is Interesting

1. **Beyond Black-Box ML**: Combines supervised ML risk probabilities with unsupervised PyTorch reconstruction loss and SHAP attributions, giving compliance officers clear visual explanations for every risk flag.
2. **Zero-Framework RAG Architecture**: Designed from scratch using FAISS and `SentenceTransformers` without heavy orchestration abstractions like LangChain, ensuring full transparency over vector chunking, indexing, similarity scoring, and prompt context injection.
3. **True Dual-Input Parallel Stream**: State-managed Streamlit session handling allows transaction data (CSV) and compliance policies (PDF/TXT) to be uploaded and analyzed independently in the same session without resetting state.
4. **Context-Grounded Evidence Search**: RAG queries are dynamically enriched with the selected customer's actual risk score, anomaly status, and top SHAP factors, retrieving exact policy evidence relevant to that customer.

---

## ⚠️ Limitations & Production Scope

- **Prototype Context**: LedgerMind AI is designed as an intelligent decision-support prototype for compliance officers. It does not replace human judgment or make autonomous binding financial decisions.
- **Dataset Scope**: Uses synthetic financial transaction data generated via `src/data_processing/generate_sample_data.py` to model complex risk relationships safely without exposing real PII.
- **Vector Store Storage**: Currently uses FAISS CPU local file storage (`models/vector_store/`); production deployments can be migrated to cloud vector databases (e.g. Pinecone, Qdrant).
