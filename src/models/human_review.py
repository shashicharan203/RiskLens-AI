import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional

class HumanReviewStore:
    """Persistent storage manager for Human-in-the-loop analyst risk decisions."""

    def __init__(self, file_path: str = "data/analyst_reviews.json"):
        root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
        self.file_path = os.path.join(root_dir, file_path)
        os.makedirs(os.path.dirname(self.file_path), exist_ok=True)
        self.reviews: List[Dict[str, Any]] = self._load()

    def _load(self) -> List[Dict[str, Any]]:
        if os.path.exists(self.file_path):
            try:
                with open(self.file_path, "r", encoding="utf-8") as f:
                    return json.load(f)
            except Exception as e:
                print(f"Error reading analyst reviews file {self.file_path}: {e}")
                return []
        return []

    def _save(self):
        with open(self.file_path, "w", encoding="utf-8") as f:
            json.dump(self.reviews, f, indent=2)

    def record_review(
        self,
        transaction_id: str,
        transaction_details: Dict[str, Any],
        risk_score: float,
        anomaly_score: float,
        decision: str, # APPROVE, REJECT, INVESTIGATE
        comments: str,
        analyst_id: str = "Analyst_01",
        shap_factors: Optional[List[str]] = None,
        rag_evidence: Optional[List[Dict[str, Any]]] = None,
        news_sentiment: Optional[str] = "Neutral",
        ai_recommendations: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """Record and persist human analyst decision."""
        if decision.upper() not in ["APPROVE", "REJECT", "INVESTIGATE"]:
            raise ValueError("Decision must be one of: APPROVE, REJECT, INVESTIGATE")
            
        record = {
            "review_id": f"REV_{len(self.reviews) + 1:04d}",
            "transaction_id": transaction_id,
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "analyst_id": analyst_id,
            "decision": decision.upper(),
            "comments": comments,
            "risk_score": risk_score,
            "anomaly_score": anomaly_score,
            "transaction_details": transaction_details,
            "shap_factors": shap_factors or [],
            "rag_evidence": rag_evidence or [],
            "news_sentiment": news_sentiment or "Neutral",
            "ai_recommendations": ai_recommendations or []
        }
        
        self.reviews.append(record)
        self._save()
        return record

    def get_all_reviews(self) -> List[Dict[str, Any]]:
        """Return all recorded analyst reviews."""
        return self.reviews

    def get_reviews_by_decision(self, decision: str) -> List[Dict[str, Any]]:
        """Filter reviews by decision."""
        return [r for r in self.reviews if r.get("decision") == decision.upper()]
