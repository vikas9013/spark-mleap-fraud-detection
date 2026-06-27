import pandas as pd
import numpy as np

def generate_transactions(n=10000):
    np.random.seed(42)
    
    # Generate features
    transaction_ids = np.arange(1, n + 1)
    
    # Normal amounts vs some large amounts for fraud in INR (approx 80x USD scale)
    amounts = np.concatenate([
        np.random.normal(4000, 1600, int(n * 0.9)),
        np.random.normal(40000, 16000, int(n * 0.1))
    ])
    amounts = np.abs(amounts)
    
    merchants = ['Groceries', 'Electronics', 'Clothing', 'Restaurants', 'Travel', 'OnlineRetail']
    merchant_categories = np.random.choice(merchants, n, p=[0.3, 0.1, 0.2, 0.2, 0.05, 0.15])
    
    devices = ['Mobile', 'Desktop', 'Tablet']
    device_types = np.random.choice(devices, n, p=[0.7, 0.25, 0.05])
    
    time_of_day = np.random.randint(0, 24, n)
    
    # Simple rule-based logic to create some correlation for the ML model to learn
    is_fraud = np.zeros(n, dtype=int)
    for i in range(n):
        fraud_prob = 0.01 # Base probability
        
        if amounts[i] > 25000:
            fraud_prob += 0.3
        if merchant_categories[i] == 'Electronics' and amounts[i] > 8000:
            fraud_prob += 0.2
        if time_of_day[i] >= 2 and time_of_day[i] <= 5: # Late night
            fraud_prob += 0.2
            
        is_fraud[i] = 1 if np.random.random() < fraud_prob else 0
        
    df = pd.DataFrame({
        'transaction_id': transaction_ids,
        'amount': amounts,
        'merchant_category': merchant_categories,
        'device_type': device_types,
        'time_of_day': time_of_day,
        'is_fraud': is_fraud
    })
    
    # Ensure there are both classes
    print(f"Total Fraudulent Transactions: {df['is_fraud'].sum()} out of {n}")
    
    df.to_csv('transactions.csv', index=False)
    print("Dataset saved to transactions.csv")

if __name__ == '__main__':
    generate_transactions()
