from typing import Dict, Any
from src.models.predict import RiskPredictor

class WhatIfSimulator:
    """Scenario analysis simulator for financial risk counterfactuals."""

    def __init__(self, model_path: str = "models/risk_model.pkl"):
        self.predictor = RiskPredictor(model_path=model_path)

    def simulate_scenario(
        self,
        base_transaction: Dict[str, Any],
        modified_params: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Simulate before vs after risk scores under parameter modifications."""
        
        # 1. Baseline prediction
        base_result = self.predictor.predict_single(base_transaction)
        base_score = base_result['risk_score']
        base_level = base_result['risk_level']
        
        # 2. Scenario prediction
        scenario_transaction = dict(base_transaction)
        scenario_transaction.update(modified_params)
        
        scenario_result = self.predictor.predict_single(scenario_transaction)
        scenario_score = scenario_result['risk_score']
        scenario_level = scenario_result['risk_level']
        
        # 3. Calculate Deltas and Impact Rationale
        score_diff = round(scenario_score - base_score, 2)
        pct_change = round(((scenario_score - base_score) / max(base_score, 0.01)) * 100, 1)
        
        impact_reasons = []
        if 'interest_rate' in modified_params:
            old_ir = base_transaction.get('interest_rate', 8.0)
            new_ir = modified_params['interest_rate']
            if new_ir > old_ir:
                impact_reasons.append("higher benchmark interest rates increasing borrower debt repayment obligation stress")
            elif new_ir < old_ir:
                impact_reasons.append("lower interest rates easing debt coverage burden")
                
        if 'transaction_amount' in modified_params:
            old_amt = base_transaction.get('transaction_amount', 500.0)
            new_amt = modified_params['transaction_amount']
            if new_amt > old_amt:
                impact_reasons.append("large transfer size exceeding baseline account spending pattern")
                
        if 'debt_to_income' in modified_params:
            if modified_params['debt_to_income'] > base_transaction.get('debt_to_income', 0.3):
                impact_reasons.append("elevated debt ratio reducing net credit liquidity buffer")

        if not impact_reasons:
            if score_diff > 0:
                impact_reasons.append("counterfactual parameter adjustment elevating overall credit risk profile")
            else:
                impact_reasons.append("favorable parameter changes mitigating default probability")

        impact_text = f"Risk {'increased' if score_diff >= 0 else 'decreased'} due to " + " and ".join(impact_reasons) + "."
        
        formatted_summary = (
            f"Current:\n"
            f"Risk Score: {int(base_score * 100)}%\n\n"
            f"Scenario:\n"
            f"New Risk Score: {int(scenario_score * 100)}%\n\n"
            f"Impact:\n"
            f"{impact_text}"
        )

        return {
            "baseline": {
                "params": base_transaction,
                "risk_score": base_score,
                "risk_level": base_level
            },
            "scenario": {
                "params": scenario_transaction,
                "risk_score": scenario_score,
                "risk_level": scenario_level
            },
            "delta": {
                "score_difference": score_diff,
                "percentage_change": pct_change,
                "direction": "INCREASED" if score_diff > 0 else ("DECREASED" if score_diff < 0 else "UNCHANGED")
            },
            "impact_summary": impact_text,
            "formatted_output": formatted_summary
        }
