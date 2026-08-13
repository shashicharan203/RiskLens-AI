import os
import random
import numpy as np
import pandas as pd

def generate_transactions(num_samples: int = 2500, random_seed: int = 42) -> pd.DataFrame:
    """Generate realistic financial transaction data with fraud/risk flags."""
    np.random.seed(random_seed)
    random.seed(random_seed)
    
    merchant_categories = [
        'Grocery', 'Utilities', 'Electronics', 'Luxury_Goods', 
        'Crypto_Exchange', 'Wire_Transfer', 'Gambling', 'Travel'
    ]
    locations = [
        'Domestic', 'International_LowRisk', 'International_HighRisk', 'Online_Unverified'
    ]
    
    data = []
    for i in range(num_samples):
        txn_id = f"TXN_{i+10001:05d}"
        acc_id = f"ACC_{random.randint(100, 999):03d}"
        
        # Base probabilities
        category = random.choices(
            merchant_categories, 
            weights=[0.35, 0.20, 0.15, 0.10, 0.05, 0.05, 0.05, 0.05]
        )[0]
        
        location = random.choices(
            locations,
            weights=[0.60, 0.20, 0.10, 0.10]
        )[0]
        
        time_hour = random.randint(0, 23)
        is_night = 1 if (time_hour < 6 or time_hour > 22) else 0
        is_weekend = 1 if random.random() < 0.28 else 0
        
        # Risk factors synthetic construction
        if category in ['Crypto_Exchange', 'Wire_Transfer', 'Gambling'] or location == 'International_HighRisk':
            base_risk_prob = 0.55
            amount = float(np.random.exponential(scale=3500) + 500)
            frequency = random.randint(8, 35)
            failed_logins = random.randint(1, 6)
            device_risk = round(float(np.random.uniform(0.6, 0.99)), 2)
            credit_util = round(float(np.random.uniform(0.65, 0.98)), 2)
            dti = round(float(np.random.uniform(0.45, 0.85)), 2)
        else:
            base_risk_prob = 0.05
            amount = float(np.random.exponential(scale=120) + 15)
            frequency = random.randint(1, 8)
            failed_logins = random.choices([0, 1, 2], weights=[0.85, 0.10, 0.05])[0]
            device_risk = round(float(np.random.uniform(0.01, 0.40)), 2)
            credit_util = round(float(np.random.uniform(0.10, 0.60)), 2)
            dti = round(float(np.random.uniform(0.15, 0.50)), 2)
            
        income = float(np.random.normal(6500, 2000))
        income = max(1500.0, round(income, 2))
        
        interest_rate = round(float(np.random.uniform(4.0, 16.5)), 2)
        account_age = random.randint(30, 3650)
        
        # High risk threshold logic
        risk_score_raw = (
            (amount / 5000) * 0.25 +
            (frequency / 30) * 0.20 +
            (1.0 if location == 'International_HighRisk' else 0.0) * 0.20 +
            (1.0 if category in ['Crypto_Exchange', 'Wire_Transfer', 'Gambling'] else 0.0) * 0.20 +
            device_risk * 0.15
        )
        is_risk = 1 if (risk_score_raw > 0.45 or random.random() < base_risk_prob * 0.4) else 0
        
        data.append({
            'transaction_id': txn_id,
            'account_id': acc_id,
            'transaction_amount': round(amount, 2),
            'transaction_frequency': frequency,
            'merchant_category': category,
            'location': location,
            'time_pattern': time_hour,
            'is_night_transaction': is_night,
            'is_weekend': is_weekend,
            'account_age_days': account_age,
            'avg_monthly_income': income,
            'debt_to_income': dti,
            'interest_rate': interest_rate,
            'credit_utilization': credit_util,
            'failed_login_attempts': failed_logins,
            'device_risk_score': device_risk,
            'is_risk': is_risk
        })
        
    df = pd.DataFrame(data)
    return df

def generate_news_data() -> pd.DataFrame:
    """Generate financial news dataset for sentiment and entity analysis."""
    news_items = [
        {
            "article_id": "NEWS_001",
            "title": "Apex Financial Group Faces Liquidity Crunch Amid Rising Debt Obligations",
            "content": "Apex Financial Group announced a quarterly loss of $45M as non-performing assets rose 18%. Executives cited high interest rates and over-leveraged corporate loans as key drivers of distress.",
            "company": "Apex Financial Group",
            "published_date": "2026-03-15"
        },
        {
            "article_id": "NEWS_002",
            "title": "Global Tech Corp Reports Record Revenue Growth and High Operating Margins",
            "content": "Global Tech Corp surpassed Wall Street estimates with a 24% year-over-year revenue increase driven by enterprise cloud sales and strong cash reserves.",
            "company": "Global Tech Corp",
            "published_date": "2026-03-18"
        },
        {
            "article_id": "NEWS_003",
            "title": "Regulatory Audit Uncovers Suspicious Wire Transfer Patterns at Zenith Capital",
            "content": "Federal regulators have launched a compliance investigation into Zenith Capital following anomalous international transactions and potential AML reporting failures.",
            "company": "Zenith Capital",
            "published_date": "2026-03-20"
        },
        {
            "article_id": "NEWS_004",
            "title": "Vanguard Retail Credit Default Rates Spike in Subprime Segment",
            "content": "Credit delinquency metrics for subprime borrowers increased by 3.2 percentage points, forcing financial institutions to tighten underwriting standards and increase capital reserves.",
            "company": "Vanguard Retail",
            "published_date": "2026-03-22"
        },
        {
            "article_id": "NEWS_005",
            "title": "Horizon Energy Secures $500M Refinancing Deal at Favorable Coupon Rates",
            "content": "Horizon Energy completed a syndicated loan restructuring deal with tier-1 investment banks, substantially lowering debt service costs and improving debt coverage ratios.",
            "company": "Horizon Energy",
            "published_date": "2026-03-25"
        }
    ]
    return pd.DataFrame(news_items)

def generate_sample_document(output_dir: str):
    """Generate sample annual report text document for RAG indexing."""
    os.makedirs(output_dir, exist_ok=True)
    file_path = os.path.join(output_dir, "annual_risk_report_2025.txt")
    
    content = """
================================================================================
LEDGERMIND FINANCIAL ANNUAL RISK & LIQUIDITY ASSESSMENT REPORT 2025
================================================================================

1. EXECUTIVE SUMMARY & RISK PROFILE
LedgerMind Financial Services maintains a rigorous credit risk monitoring framework across commercial and retail portfolios. In FY2025, total managed credit exposure reached $4.2 Billion. High-risk transactions are defined as those exhibiting unexpected velocity (>10 transactions per hour), high debt-to-income ratios (>0.50), high credit utilization (>80%), or cross-border transfers to unverified offshore entities.

2. LIQUIDITY AND DEBT OBLIGATIONS
The corporate loan portfolio experienced a modest increase in non-performing loans (NPL) from 1.8% to 3.4%. Higher benchmark interest rates (currently 8.5%) have elevated default risk for highly leveraged borrowers. Debt-to-income ratio thresholds have been lowered to 0.45 to prevent systemic credit defaults.

3. TRANSACTIONAL ANOMALY INDICATORS
Our Quantitative Risk Engine identifies four primary anomaly vectors:
- Rapid Velocity: Multiple high-value transfers within a 15-minute window.
- Off-Hours Settlement: Large transfers executed between 22:00 and 05:00 local time.
- Unverified Foreign Destinations: Direct wires to high-risk international jurisdictions.
- Device Anomaly: Authentication attempts from unverified IP ranges coupled with multiple failed login attempts.

4. CREDIT UNDERWRITING POLICY & RECOMMENDATIONS
To mitigate elevated default and fraud exposure:
1. Require multi-factor step-up authentication for any transfer exceeding $5,000 to new counterparties.
2. Freeze accounts experiencing more than 3 failed login attempts within 10 minutes.
3. Automatically elevate customer risk tier to 'HIGH' if credit utilization exceeds 85% and debt-to-income exceeds 0.60.
4. Conduct quarterly portfolio stress tests under simulated benchmark interest rate increases (+200 bps).

5. REGULATORY COMPLIANCE AND ANTI-MONEY LAUNDERING (AML)
All international wire transfers over $10,000 undergo automated suspicious activity reporting (SAR) screening. Entities operating in high-friction merchant categories (Crypto Exchanges, Gambling, Unregulated Forex) are subjected to continuous transaction monitoring and enhanced due diligence.
"""
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(content.strip())
    print(f"Sample annual report created at {file_path}")

def main():
    root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
    data_dir = os.path.join(root_dir, "data")
    docs_dir = os.path.join(data_dir, "documents")
    
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(docs_dir, exist_ok=True)
    
    df_tx = generate_transactions(num_samples=2500)
    tx_path = os.path.join(data_dir, "transactions.csv")
    df_tx.to_csv(tx_path, index=False)
    print(f"Generated {len(df_tx)} transaction records at {tx_path}")
    print(f"Risk distribution: {df_tx['is_risk'].value_counts().to_dict()}")
    
    df_news = generate_news_data()
    news_path = os.path.join(data_dir, "financial_news.csv")
    df_news.to_csv(news_path, index=False)
    print(f"Generated {len(df_news)} news articles at {news_path}")
    
    generate_sample_document(docs_dir)

if __name__ == "__main__":
    main()
