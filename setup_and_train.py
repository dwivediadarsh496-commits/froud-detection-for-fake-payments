import pandas as pd
import numpy as np
import joblib
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LogisticRegression

def generate_synthetic_data(filename="AIML Dataset.csv", num_rows=5000):
    print("Generating synthetic dataset...")
    np.random.seed(42)
    
    types = ['PAYMENT', 'TRANSFER', 'CASH_OUT', 'CASH_IN', 'DEBIT']
    
    # Generate random features matching the PaySim distribution schema
    step = np.random.randint(1, 100, size=num_rows)
    tx_type = np.random.choice(types, size=num_rows)
    amount = np.random.exponential(scale=20000, size=num_rows) + 10.0
    
    # Generate realistic names
    nameOrig = [f"C{np.random.randint(100000000, 999999999)}" for _ in range(num_rows)]
    nameDest = [f"M{np.random.randint(100000000, 999999999)}" if t == 'PAYMENT' else f"C{np.random.randint(100000000, 999999999)}" for t in tx_type]
    
    oldbalanceOrg = np.random.exponential(scale=100000, size=num_rows)
    # Deduct amount for originated transaction
    newbalanceOrig = np.clip(oldbalanceOrg - amount, 0, None)
    
    oldbalanceDest = np.random.exponential(scale=200000, size=num_rows)
    newbalanceDest = oldbalanceDest + amount
    
    # Simulate fraud condition: Transfer/Cash-out with high amounts
    isFraud = np.zeros(num_rows, dtype=int)
    fraud_indices = np.random.choice(num_rows, size=int(num_rows * 0.05), replace=False)
    isFraud[fraud_indices] = 1
    
    # For fraud rows, adjust amounts to be high and make balances match
    for idx in fraud_indices:
        amount[idx] = np.random.uniform(100000, 1000000)
        tx_type[idx] = np.random.choice(['TRANSFER', 'CASH_OUT'])
        newbalanceOrig[idx] = 0.0
    
    isFlaggedFraud = np.zeros(num_rows, dtype=int)
    isFlaggedFraud[(isFraud == 1) & (amount > 200000)] = 1

    df = pd.DataFrame({
        'step': step,
        'type': tx_type,
        'amount': amount,
        'nameOrig': nameOrig,
        'oldbalanceOrg': oldbalanceOrg,
        'newbalanceOrig': newbalanceOrig,
        'nameDest': nameDest,
        'oldbalanceDest': oldbalanceDest,
        'newbalanceDest': newbalanceDest,
        'isFraud': isFraud,
        'isFlaggedFraud': isFlaggedFraud
    })
    
    df.to_csv(filename, index=False)
    print(f"Saved synthetic dataset to {filename} with {num_rows} rows.")
    return df

def train_model(df):
    print("Training model pipeline...")
    # Select features
    features = ['type', 'amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    X = df[features]
    y = df['isFraud']
    
    num_features = ['amount', 'oldbalanceOrg', 'newbalanceOrig', 'oldbalanceDest', 'newbalanceDest']
    cat_features = ['type']
    
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), num_features),
            ('cat', OneHotEncoder(drop='first'), cat_features)
        ]
    )
    
    model_pipeline = Pipeline(
        steps=[
            ('preprocessor', preprocessor),
            ('classifier', LogisticRegression(class_weight='balanced', max_iter=1000))
        ]
    )
    
    model_pipeline.fit(X, y)
    joblib.dump(model_pipeline, "fraud_detection_model.pkl")
    print("Model pipeline successfully trained and saved as 'fraud_detection_model.pkl'.")

if __name__ == '__main__':
    df = generate_synthetic_data()
    train_model(df)
