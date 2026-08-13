from typing import Dict, Any, List

class RiskAdvisorEngine:
    """AI Recommendation Engine for financial risk mitigation."""

    def generate_recommendations(
        self,
        risk_score: float,
        shap_factors: List[str] = None,
        retrieved_evidence: List[Dict[str, Any]] = None,
        news_sentiment: str = "Neutral"
    ) -> Dict[str, Any]:
        """Generate tailored financial risk mitigation recommendations."""
        
        recommendations = []
        shap_factors = shap_factors or []
        
        # 1. Base Score Rules
        if risk_score >= 0.70:
            recommendations.append("Increase transaction monitoring and freeze unverified wire transfers.")
            recommendations.append("Perform immediate step-up multi-factor authentication (MFA) verification.")
            recommendations.append("Reduce credit exposure and lower daily transaction limits.")
            recommendations.append("Trigger automated Suspicious Activity Report (SAR) compliance review.")
        elif risk_score >= 0.35:
            recommendations.append("Review account activity for unusual velocity trends.")
            recommendations.append("Require additional customer identity verification.")
            recommendations.append("Monitor account for next 48 hours for repeated failed authentication attempts.")
        else:
            recommendations.append("Maintain standard automated monitoring protocols.")
            recommendations.append("No immediate account restriction required.")

        # 2. SHAP Factor Specific Advice
        factors_str = " ".join(shap_factors).lower()
        if "amount" in factors_str:
            recommendations.append("Enforce temporary holds on high-value single transfers exceeding $5,000.")
        if "location" in factors_str:
            recommendations.append("Block foreign IP ranges originating from high-risk international jurisdictions.")
        if "merchant" in factors_str or "crypto" in factors_str or "gambling" in factors_str:
            recommendations.append("Restrict high-friction merchant category transaction authorizations.")
        if "debt" in factors_str or "credit" in factors_str:
            recommendations.append("Initiate debt refinancing review and assess borrower credit utilization limit.")
        if "failed login" in factors_str or "device" in factors_str:
            recommendations.append("Reset customer security credentials and revoke unrecognized active device sessions.")

        # 3. News Sentiment Rule
        if news_sentiment == "Negative":
            recommendations.append("Conduct enhanced due diligence regarding adverse news media alerts for entity.")

        # Deduplicate and pick top recommendations
        unique_recs = list(dict.fromkeys(recommendations))
        final_recs = unique_recs[:4]
        
        formatted_text = "Recommendations:\n\n" + "\n".join([f"{idx+1}. {rec}" for idx, rec in enumerate(final_recs)])
        
        return {
            "risk_score": risk_score,
            "recommendations": final_recs,
            "recommendations_formatted": formatted_text
        }
