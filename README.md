# 💳 AI Fraud Detection System for Fake Payments

[![Live Demo](https://img.shields.io/badge/Live%20Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit)](https://froud-detection-for-fake-payments-olithhybrdmgxrcvcxmrpn.streamlit.app/)


> A machine learning-powered fraud detection dashboard for identifying fake payment transactions, built with Streamlit and scikit-learn. Trained on a large-scale AIML dataset of over 6.3 million transactions.

---

## 🚀 Live Demo

👉 **[Open the Dashboard](https://froud-detection-for-fake-payments-olithhybrdmgxrcvcxmrpn.streamlit.app/)**

---

## 📖 Overview

This project detects fraudulent financial transactions using a trained machine learning model. It provides an interactive Streamlit dashboard with four views: dataset overview, exploratory data analysis (EDA), real-time fraud prediction, and a transaction network graph.

---

## ✨ Features

- **📊 Dataset Overview** — See total transactions, fraud cases, and fraud rate at a glance, with a preview of the raw data.
- **📈 EDA (Exploratory Data Analysis)** — Bar charts of transaction type distribution, fraud rate by type, and log-scale amount histogram.
- **🔍 Fraud Prediction** — Enter transaction details (amount, sender/receiver balances, type) and instantly get a fraud prediction with probability score.
- **🕸️ Transaction Network Graph** — Visualize transaction flows as a directed graph, with fraudulent edges highlighted in red using NetworkX.

---

## 🗂️ Project Structure

```
froud-detection-for-fake-payments/
├── AIML Dataset.csv              # Training dataset (~6.3M transactions)
├── app.ipynb                     # Jupyter notebook for model training & EDA
├── setup_and_train.py            # Script to preprocess data and train the model
├── fraud_detection_model.pkl     # Saved trained ML model (joblib)
├── streamlit_fraud.py            # Main Streamlit app
├── requirements.txt              # Python dependencies
└── .gitignore
```

---

## 🧠 Model

The model is trained using scikit-learn on the AIML Dataset with the following transaction features:

| Feature | Description |
|---|---|
| `type` | Transaction type (CASH_OUT, TRANSFER, etc.) |
| `amount` | Transaction amount |
| `oldbalanceOrg` | Sender's balance before transaction |
| `newbalanceOrig` | Sender's balance after transaction |
| `oldbalanceDest` | Receiver's balance before transaction |
| `newbalanceDest` | Receiver's balance after transaction |

The trained model is persisted as `fraud_detection_model.pkl` via `joblib` for fast loading in the app.

---

## ⚙️ Installation & Local Setup

**1. Clone the repository**
```bash
git clone https://github.com/dwivediadarsh496-commits/froud-detection-for-fake-payments.git
cd froud-detection-for-fake-payments
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Train the model** (if `.pkl` not present)
```bash
python setup_and_train.py
```

**4. Run the Streamlit app**
```bash
streamlit run streamlit_fraud.py
```

---

## 📦 Requirements

```
streamlit
pandas
scikit-learn
joblib
networkx
matplotlib
seaborn
```

---

## 📊 Dataset

The project uses the **AIML Dataset** (`AIML Dataset.csv`), a large-scale synthetic financial transactions dataset containing over 6 million rows with labeled fraud cases (`isFraud` column). The dataset includes fields like transaction type, originator/destination account names and balances, and fraud labels.

> **Note:** Due to file size, you may need to download the dataset separately and place it in the project root as `AIML Dataset.csv`.

---

## 🖥️ Dashboard Views

### Overview
Displays key metrics (total transactions, fraud count, fraud percentage) and a sample of the raw dataframe.

### EDA
- Transaction type distribution (bar chart)
- Fraud rate per transaction type (bar chart)
- Log-scale amount distribution (histogram with KDE)

### Fraud Detection
Enter transaction parameters and click **Predict Fraud** to get:
- 🚨 `FRAUD DETECTED` with probability score
- ✅ `SAFE TRANSACTION` with probability score

### Graph View
An interactive directed graph of sampled transactions, where:
- **Gray edges** = normal transactions
- **Red edges** = fraudulent transactions
- Adjustable sample size (50–300 nodes) via slider

---

## 🚀 Deployment

The app is deployed on **Streamlit Community Cloud**.

To deploy your own instance:
1. Push the repo to GitHub (with `fraud_detection_model.pkl` and `AIML Dataset.csv`)
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set the main file to `streamlit_fraud.py`

---

## 🛠️ Tech Stack

| Tool | Purpose |
|---|---|
| Python | Core language |
| Streamlit | Web dashboard framework |
| scikit-learn | ML model training & inference |
| pandas / NumPy | Data manipulation |
| Matplotlib / Seaborn | Data visualization |
| NetworkX | Transaction network graph |
| joblib | Model serialization |

---

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

---

## 👤 Author

**dwivediadarsh496-commits**
- GitHub: [@dwivediadarsh496-commits](https://github.com/dwivediadarsh496-commits)

---

> 🚀 Built with AI + Graph Intelligence
