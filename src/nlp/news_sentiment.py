import re
from typing import Dict, Any, List

class FinancialNewsAnalyzer:
    """Financial news NLP pipeline using FinBERT for sentiment and entity impact extraction (Lazy-loaded)."""

    def __init__(self, model_name: str = "ProsusAI/finbert"):
        self.model_name = model_name
        self.sentiment_pipeline = None
        self._initialized = False

    def _get_pipeline(self):
        """Lazy load FinBERT transformers pipeline on demand."""
        if not self._initialized:
            self._initialized = True
            try:
                from transformers import pipeline
                self.sentiment_pipeline = pipeline(
                    "text-classification",
                    model=self.model_name,
                    return_all_scores=False
                )
            except Exception as e:
                print(f"Notice: FinBERT pipeline deferred or memory constrained ({e}). Using financial lexicon sentiment engine.")
                self.sentiment_pipeline = None
        return self.sentiment_pipeline

    def extract_entities(self, text: str) -> List[str]:
        """Extract financial corporate entities from news text."""
        known_companies = [
            "Apex Financial Group", "Global Tech Corp", "Zenith Capital",
            "Vanguard Retail", "Horizon Energy", "JPMorgan", "Goldman Sachs",
            "Bank of America", "Citigroup", "Morgan Stanley"
        ]
        found = [comp for comp in known_companies if comp.lower() in text.lower()]
        
        if not found:
            # Fallback regex matcher for capitalized corporate names
            matches = re.findall(r'\b[A-Z][a-z]+ (?:Financial|Capital|Corp|Inc|Bank|Group|Energy|Holdings)\b', text)
            found = list(set(matches)) if matches else ["Target Enterprise"]
            
        return found

    def evaluate_impact(self, sentiment: str, text: str) -> str:
        """Map sentiment and text context to financial risk impact."""
        text_lower = text.lower()
        high_risk_keywords = ["debt", "default", "investigation", "liquidity", "loss", "aml", "fraud", "delinquency", "audit"]
        
        if sentiment.lower() == "negative":
            if any(k in text_lower for k in high_risk_keywords):
                return "High Risk"
            return "Medium Risk"
        elif sentiment.lower() == "neutral":
            return "Low Risk"
        else: # Positive
            return "Low Risk"

    def analyze_news(self, text: str) -> Dict[str, Any]:
        """Analyze a financial news article and extract Sentiment, Entity, and Risk Impact."""
        sentiment_label = "Neutral"
        confidence = 0.85
        
        pipe = self._get_pipeline()
        if pipe is not None:
            try:
                res = pipe(text[:512])[0]
                sentiment_label = res['label'].capitalize()
                confidence = float(res['score'])
            except Exception:
                sentiment_label = self._fallback_sentiment(text)
        else:
            sentiment_label = self._fallback_sentiment(text)
            
        entities = self.extract_entities(text)
        impact = self.evaluate_impact(sentiment_label, text)
        
        return {
            "sentiment": sentiment_label,
            "confidence": round(confidence, 4),
            "entity": entities[0] if entities else "Company",
            "entities": entities,
            "impact": impact
        }

    def _fallback_sentiment(self, text: str) -> str:
        """Financial lexicon sentiment fallback."""
        text_lower = text.lower()
        neg_words = ["loss", "debt", "decline", "investigation", "default", "audit", "fraud", "crunch", "drop"]
        pos_words = ["growth", "record", "revenue", "profit", "surpassed", "refinancing", "gain", "expanded"]
        
        neg_count = sum(1 for w in neg_words if w in text_lower)
        pos_count = sum(1 for w in pos_words if w in text_lower)
        
        if neg_count > pos_count:
            return "Negative"
        elif pos_count > neg_count:
            return "Positive"
        else:
            return "Neutral"
