import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
import networkx as nx
import os

st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")

st.title("💳 AI Fraud Detection System")

# Cache data loading to optimize performance for large datasets (6.3M rows)
@st.cache_data
def load_data(filepath):
    if not os.path.exists(filepath):
        return None
    return pd.read_csv(filepath)

# Cache model loading
@st.cache_resource
def load_model(filepath):
    if not os.path.exists(filepath):
        return None
    return joblib.load(filepath)

df = load_data("AIML Dataset.csv")
model = load_model("fraud_detection_model.pkl")

# Error handling if files are missing
if df is None:
    st.error("🚨 **Dataset 'AIML Dataset.csv' not found.** Please make sure the dataset is placed in the project directory.")
    st.stop()

if model is None:
    st.error("🚨 **Model file 'fraud_detection_model.pkl' not found.** Please make sure the model is trained and saved in the project directory.")
    st.stop()

st.sidebar.header("⚙️ Controls")

option = st.sidebar.selectbox(
    "Select View",
    ["Overview", "EDA", "Fraud Detection", "Graph View"]
)

if option == "Overview":
    st.subheader("📊 Dataset Overview")

    col1, col2, col3 = st.columns(3)

    col1.metric("Total Transactions", len(df))
    col2.metric("Fraud Cases", df['isFraud'].sum())
    col3.metric("Fraud %", f"{round(df['isFraud'].mean()*100,2)}%")

    st.dataframe(df.head())

elif option == "EDA":
    st.subheader("📈 Exploratory Data Analysis")

    col1, col2 = st.columns(2)

    with col1:
        st.write("Transaction Types")
        fig, ax = plt.subplots()
        df['type'].value_counts().plot(kind='bar', ax=ax)
        st.pyplot(fig)

    with col2:
        st.write("Fraud Rate by Type")
        fig, ax = plt.subplots()
        df.groupby('type')['isFraud'].mean().plot(kind='bar', ax=ax)
        st.pyplot(fig)

    st.write("Amount Distribution")
    fig, ax = plt.subplots()
    # Sample dataset for faster plotting and responsiveness
    sample_amount = df['amount'].sample(min(100000, len(df)), random_state=42)
    sns.histplot(np.log1p(sample_amount), bins=50, kde=True, ax=ax)
    st.pyplot(fig)

elif option == "Fraud Detection":
    st.subheader("🔍 Predict Transaction Fraud")

    amount = st.number_input("Amount", value=1000.0)
    oldbalanceOrg = st.number_input("Old Balance (Sender)", value=5000.0)
    newbalanceOrig = st.number_input("New Balance (Sender)", value=4000.0)
    oldbalanceDest = st.number_input("Old Balance (Receiver)", value=0.0)
    newbalanceDest = st.number_input("New Balance (Receiver)", value=1000.0)
    tx_type = st.selectbox("Transaction Type", df['type'].unique())

    if st.button("Predict Fraud"):
        input_df = pd.DataFrame([{
            'type': tx_type,
            'amount': amount,
            'oldbalanceOrg': oldbalanceOrg,
            'newbalanceOrig': newbalanceOrig,
            'oldbalanceDest': oldbalanceDest,
            'newbalanceDest': newbalanceDest
        }])

        prediction = model.predict(input_df)[0]
        prob = model.predict_proba(input_df)[0][1]

        if prediction == 1:
            st.error(f"🚨 FRAUD DETECTED (Probability: {prob:.2f})")
        else:
            st.success(f"✅ SAFE TRANSACTION (Probability: {prob:.2f})")

elif option == "Graph View":
    st.subheader("🕸️ Transaction Network")

    sample_size = st.slider("Select sample size", 50, 300, 100)

    # Smart sampling: ensure both normal and fraud cases are represented in the network graph
    df_fraud = df[df['isFraud'] == 1]
    df_normal = df[df['isFraud'] == 0]

    n_fraud = min(len(df_fraud), int(sample_size * 0.15))  # target 15% fraud in visualization
    n_normal = sample_size - n_fraud

    df_sample = pd.concat([
        df_fraud.sample(n_fraud, random_state=42) if n_fraud > 0 else pd.DataFrame(),
        df_normal.sample(n_normal, random_state=42)
    ]).sample(frac=1, random_state=42)

    G = nx.DiGraph()

    for _, row in df_sample.iterrows():
        G.add_edge(
            row['nameOrig'],
            row['nameDest'],
            isFraud=row['isFraud']
        )

    pos = nx.spring_layout(G, k=0.4)

    fig, ax = plt.subplots(figsize=(8,6))

    fraud_edges = [(u,v) for u,v,d in G.edges(data=True) if d['isFraud']==1]
    normal_edges = [(u,v) for u,v,d in G.edges(data=True) if d['isFraud']==0]

    nx.draw_networkx_nodes(G, pos, node_size=50, ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=normal_edges, edge_color='gray', ax=ax)
    nx.draw_networkx_edges(G, pos, edgelist=fraud_edges, edge_color='red', width=2, ax=ax)

    ax.set_title("Red = Fraud Transactions")
    ax.axis('off')

    st.pyplot(fig)

st.markdown("---")
st.markdown("🚀 Built with AI + Graph Intelligence")