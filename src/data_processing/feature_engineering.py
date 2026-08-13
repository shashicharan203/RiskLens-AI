import pandas as pd
import numpy as np

class FeatureEngineer:
    """Feature engineering pipeline for financial transaction risk detection."""

    @staticmethod
    def add_engineered_features(df: pd.DataFrame) -> pd.DataFrame:
        """Derive risk interactions, ratios, and domain features."""
        df_out = df.copy()

        # 1. Amount to Income Ratio
        if 'transaction_amount' in df_out.columns and 'avg_monthly_income' in df_out.columns:
            income_safe = df_out['avg_monthly_income'].replace(0, 1.0)
            df_out['amount_to_income_ratio'] = df_out['transaction_amount'] / income_safe

        # 2. High Risk Category Indicator
        high_risk_cats = ['Crypto_Exchange', 'Wire_Transfer', 'Gambling']
        if 'merchant_category' in df_out.columns:
            df_out['is_high_risk_merchant'] = df_out['merchant_category'].isin(high_risk_cats).astype(int)

        # 3. High Risk Location Indicator
        if 'location' in df_out.columns:
            df_out['is_high_risk_location'] = (df_out['location'] == 'International_HighRisk').astype(int)

        # 4. Debt & Credit Stress Interaction
        if 'debt_to_income' in df_out.columns and 'credit_utilization' in df_out.columns:
            df_out['credit_debt_stress_score'] = df_out['debt_to_income'] * df_out['credit_utilization']

        # 5. Night Anomaly Interaction
        if 'is_night_transaction' in df_out.columns and 'transaction_amount' in df_out.columns:
            df_out['night_amount_impact'] = df_out['is_night_transaction'] * np.log1p(df_out['transaction_amount'])

        # 6. Auth Anomaly Score
        if 'failed_login_attempts' in df_out.columns and 'device_risk_score' in df_out.columns:
            df_out['auth_anomaly_score'] = df_out['failed_login_attempts'] * df_out['device_risk_score']

        return df_out
