# 💳 Financial Transaction Risk Scorer

## 🧠 Overview

The **Financial Transaction Risk Scorer** is a machine learning project built to detect and score potential fraudulent financial transactions in real time.  
It uses advanced classification algorithms to identify high-risk transactions, helping financial institutions prevent fraud losses and improve customer trust.

This system simulates how real-world banks, fintech platforms, and payment gateways use AI to make instant fraud decisions.

---

## 📂 Dataset

**Source:** Synthetic yet realistic financial transaction dataset generated for research and development purposes.

| Column              | Description                                           |
| :------------------ | :---------------------------------------------------- |
| `transaction_id`    | Unique identifier for each transaction                |
| `amount`            | Transaction amount (USD)                              |
| `merchant_category` | Type of merchant or service                           |
| `country`           | Country where the transaction occurred                |
| `payment_method`    | Method of payment (credit card, PayPal, crypto, etc.) |
| `customer_age`      | Age of the customer                                   |
| `customer_tenure`   | How long the customer has been active                 |
| `risk_score`        | Calculated risk score from external rules engine      |
| `is_fraud`          | Target variable (1 = Fraudulent, 0 = Legitimate)      |

Dataset size: **15,000 records**

---

## ⚙️ Project Pipeline

### 1. Data Preparation

- Loaded CSV data using **Pandas**
- Cleaned and validated for missing or inconsistent entries
- Encoded categorical features using **OneHotEncoder**
- Scaled numerical features for balanced model learning

### 2. Model Training

Two machine learning models were trained and compared:

- **Random Forest Classifier**
- **XGBoost Classifier**

The training data was split into 75% training and 25% testing.

### 3. Evaluation Metrics

| Metric        | Random Forest | XGBoost |
| :------------ | :------------ | :------ |
| **Accuracy**  | 0.9587        | 0.9590  |
| **Precision** | 0.9913        | 0.9446  |
| **Recall**    | 0.8351        | 0.9259  |
| **F1-Score**  | 0.9065        | 0.9352  |
| **ROC AUC**   | 0.994         | 0.994   |
| **PR AUC**    | 0.988         | 0.988   |

✅ Both models achieved **exceptional accuracy (95%+)**, showing high precision with excellent fraud separation capability.

---

## 🧾 Key Insights

- Fraudulent transactions made up **31.9%** of the dataset.
- The model correctly identified **83.5%** of fraud cases while keeping false positives below **1%**.
- RandomForest achieved the best balance between precision and recall.
- SHAP explainability revealed that **amount**, **country**, and **risk_score** are the top fraud indicators.

---

## 💼 Business Impact

This model can be used by:

- **Banks** to block suspicious transactions in real time
- **Fintechs** to auto-flag high-risk payments
- **Payment gateways** to reduce chargeback losses
- **Risk teams** to generate fraud risk reports and insights

It demonstrates how AI-driven decision systems enhance **fraud prevention accuracy** while maintaining smooth user experience.

---

## 🧰 Tech Stack

- **Python 3.12**
- **Pandas**
- **Scikit-learn**
- **XGBoost**
- **Matplotlib**
- **Joblib**
- **SHAP**

---

## 🚀 How to Run

```bash
# 1️⃣ Install dependencies
pip install -r requirements.txt

# 2️⃣ Run model training
python train_risk_scorer.py

📊 Future Improvements
	•	Integrate into a FastAPI or Flask REST API for real-time fraud scoring
	•	Build a Power BI or Streamlit Dashboard to visualize risk patterns
	•	Apply SMOTE for improved handling of imbalanced datasets
	•	Deploy using Docker + Azure ML for enterprise-grade scaling

⸻

🧑‍💼 Author

Nyambura Kamau
Product-Focused Data Professional | Fraud Analytics | Risk Scoring | AI Strategy
📧 kamaurosemary.n@gmail.com
🌍 Nairobi, Kenya
```
