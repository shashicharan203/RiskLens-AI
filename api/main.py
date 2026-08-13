import os
import sys
import io
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# Ensure project root is in python path
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)

from src.models.predict import RiskPredictor
from src.explainability.shap_explainer import ShapRiskExplainer
from src.rag.retrieval import RAGRetriever
from src.llm.generator import CustomRAGGenerator
from src.nlp.news_sentiment import FinancialNewsAnalyzer
from src.recommendation.advisor import RiskAdvisorEngine
from src.simulation.what_if import WhatIfSimulator
from src.models.human_review import HumanReviewStore
from src.data_processing.input_handler import InputHandler

app = FastAPI(
    title="RiskLens AI API",
    description="Explainable Financial Risk & Compliance Intelligence Platform Backend API",
    version="1.0.0"
)

# Enable CORS for Streamlit frontend / web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global Service Singleton Container
_services: Dict[str, Any] = {}

def get_services() -> Dict[str, Any]:
    """Lazy initialize and return all platform services."""
    if not _services:
        model_path = os.path.join(root_dir, "models/risk_model.pkl")
        if not os.path.exists(model_path):
            print("Model file missing. Triggering pipeline training...")
            from src.models.train import train_and_save_pipeline
            train_and_save_pipeline()
            
        _services["predictor"] = RiskPredictor()
        _services["explainer"] = ShapRiskExplainer()
        _services["retriever"] = RAGRetriever()
        _services["generator"] = CustomRAGGenerator()
        _services["news_analyzer"] = FinancialNewsAnalyzer()
        _services["advisor"] = RiskAdvisorEngine()
        _services["simulator"] = WhatIfSimulator()
        _services["review_store"] = HumanReviewStore()
        print("All RiskLens AI services successfully loaded.")
        
    return _services

@app.on_event("startup")
def startup_event():
    print("RiskLens AI FastAPI backend started. Services will be lazy-loaded on demand.")

# --- Request / Response Models ---

class TransactionSchema(BaseModel):
    customer_id: Optional[str] = Field(default="CUST_0042", json_schema_extra={"example": "CUST_0042"})
    customer_name: Optional[str] = Field(default="Example Customer", json_schema_extra={"example": "Example Customer"})
    transaction_amount: float = Field(..., json_schema_extra={"example": 4500.0})
    transaction_frequency: int = Field(..., json_schema_extra={"example": 12})
    merchant_category: str = Field(..., json_schema_extra={"example": "Crypto_Exchange"})
    location: str = Field(..., json_schema_extra={"example": "International_HighRisk"})
    time_pattern: int = Field(default=2, json_schema_extra={"example": 2})
    is_night_transaction: int = Field(default=1, json_schema_extra={"example": 1})
    is_weekend: int = Field(default=0, json_schema_extra={"example": 0})
    account_age_days: int = Field(default=180, json_schema_extra={"example": 180})
    avg_monthly_income: float = Field(default=6000.0, json_schema_extra={"example": 6000.0})
    debt_to_income: float = Field(default=0.65, json_schema_extra={"example": 0.65})
    interest_rate: float = Field(default=8.5, json_schema_extra={"example": 8.5})
    credit_utilization: float = Field(default=0.88, json_schema_extra={"example": 0.88})
    failed_login_attempts: int = Field(default=4, json_schema_extra={"example": 4})
    device_risk_score: float = Field(default=0.85, json_schema_extra={"example": 0.85})

class EvidenceQuerySchema(BaseModel):
    query: str = Field(..., json_schema_extra={"example": "Why is this customer considered high risk according to policy?"})
    top_k: int = Field(default=4, json_schema_extra={"example": 4})
    transaction_context: Optional[Dict[str, Any]] = None

class SimulationSchema(BaseModel):
    base_transaction: Dict[str, Any]
    modified_params: Dict[str, Any]

class RecommendationSchema(BaseModel):
    risk_score: float = Field(..., json_schema_extra={"example": 0.82})
    shap_factors: Optional[List[str]] = None
    retrieved_evidence: Optional[List[Dict[str, Any]]] = None
    news_sentiment: Optional[str] = "Negative"

class NewsSchema(BaseModel):
    text: str = Field(..., json_schema_extra={"example": "Apex Financial Group faces quarterly loss as non-performing assets rise 18%."})

class ReviewSubmissionSchema(BaseModel):
    transaction_id: str = Field(default="CUST_0042")
    transaction_details: Dict[str, Any]
    risk_score: float
    anomaly_score: float
    decision: str = Field(..., json_schema_extra={"example": "REJECT"})
    comments: str = Field(..., json_schema_extra={"example": "Multiple failed login attempts and high device risk violate compliance rules."})
    analyst_id: str = Field(default="Analyst_01")
    shap_factors: Optional[List[str]] = None
    rag_evidence: Optional[List[Dict[str, Any]]] = None
    news_sentiment: Optional[str] = "Negative"
    ai_recommendations: Optional[List[str]] = None

# --- API Endpoints ---

@app.get("/health")
def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "RiskLens AI Platform API",
        "version": "1.0.0"
    }

@app.post("/predict-risk")
def predict_risk(data: TransactionSchema):
    """Predict financial transaction risk, anomaly score, combined assessment, & human review flag."""
    try:
        services = get_services()
        result = services["predictor"].predict_single(data.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/explain-risk")
def explain_risk(data: TransactionSchema):
    """Generate SHAP-based Explainable AI breakdown."""
    try:
        services = get_services()
        result = services["explainer"].explain_transaction(data.model_dump())
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/model-evaluation")
def model_evaluation():
    """Return complete comparative performance metrics for all models."""
    try:
        services = get_services()
        return {
            "all_model_metrics": services["predictor"].all_model_metrics,
            "selected_model": services["predictor"].model_type,
            "is_synthetic_data": True
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/upload-transactions-csv")
async def upload_transactions_csv(file: UploadFile = File(...)):
    """Upload, validate, preprocess, and score batch transactions CSV file."""
    try:
        contents = await file.read()
        valid, msg, df_clean = InputHandler.validate_and_clean_csv(io.BytesIO(contents))
        if not valid:
            raise HTTPException(status_code=400, detail=msg)
            
        services = get_services()
        df_scored = services["predictor"].predict_batch(df_clean)
        
        return {
            "status": "success",
            "message": msg,
            "total_transactions": len(df_scored),
            "high_risk_count": int((df_scored["risk_level"] == "HIGH").sum()),
            "predictions": df_scored.to_dict(orient="records")
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"CSV processing error: {str(e)}")

@app.post("/upload-document")
async def upload_document(file: UploadFile = File(...)):
    """Upload PDF or TXT financial report, parse, chunk, embed, and index into FAISS vector store."""
    try:
        contents = await file.read()
        services = get_services()
        
        valid, msg, chunks = InputHandler.process_uploaded_document_bytes(
            file_name=file.filename,
            file_bytes=contents,
            retriever_instance=services["retriever"]
        )
        if not valid:
            raise HTTPException(status_code=400, detail=msg)
            
        return {
            "status": "success",
            "message": msg,
            "chunks_indexed": len(chunks),
            "document_name": file.filename
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Document upload error: {str(e)}")

@app.post("/retrieve-evidence")
def retrieve_evidence(data: EvidenceQuerySchema):
    """Custom RAG retrieval and answer synthesis with evidence citations."""
    try:
        services = get_services()
        evidence_chunks = services["retriever"].retrieve(data.query, top_k=data.top_k)
        response = services["generator"].generate_response(
            query=data.query,
            retrieved_evidence=evidence_chunks,
            transaction_context=data.transaction_context
        )
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/analyze-news")
def analyze_news(data: NewsSchema):
    """Financial news FinBERT sentiment, entity & risk impact analysis."""
    try:
        services = get_services()
        result = services["news_analyzer"].analyze_news(data.text)
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/recommend")
def recommend_actions(data: RecommendationSchema):
    """AI recommendation engine for risk mitigation."""
    try:
        services = get_services()
        result = services["advisor"].generate_recommendations(
            risk_score=data.risk_score,
            shap_factors=data.shap_factors,
            retrieved_evidence=data.retrieved_evidence,
            news_sentiment=data.news_sentiment or "Neutral"
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/simulate")
def simulate_scenario(data: SimulationSchema):
    """What-if counterfactual scenario simulator."""
    try:
        services = get_services()
        result = services["simulator"].simulate_scenario(
            base_transaction=data.base_transaction,
            modified_params=data.modified_params
        )
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/review-transaction")
def review_transaction(data: ReviewSubmissionSchema):
    """Record Human-in-the-loop analyst review decision."""
    try:
        services = get_services()
        record = services["review_store"].record_review(
            transaction_id=data.transaction_id,
            transaction_details=data.transaction_details,
            risk_score=data.risk_score,
            anomaly_score=data.anomaly_score,
            decision=data.decision,
            comments=data.comments,
            analyst_id=data.analyst_id,
            shap_factors=data.shap_factors,
            rag_evidence=data.rag_evidence,
            news_sentiment=data.news_sentiment,
            ai_recommendations=data.ai_recommendations
        )
        return {"status": "success", "review_record": record}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/reviews")
def get_reviews():
    """Retrieve all recorded human analyst reviews."""
    try:
        services = get_services()
        return {"reviews": services["review_store"].get_all_reviews()}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
