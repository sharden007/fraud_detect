import pandas as pd
import numpy as np

# Set random seed for reproducibility
np.random.seed(42)

# Generate synthetic data
n_samples = 1000000  # Larger dataset
transaction_amount = np.random.uniform(1, 10000, n_samples)
location = np.random.randint(1, 100, n_samples)  # 100 unique locations
customer_id = np.random.randint(1, 1000, n_samples)  # 1000 unique customers
transaction_time = np.random.randint(0, 24, n_samples)  # Hour of the day (0-23)
device_type = np.random.choice(['mobile', 'desktop', 'tablet'], n_samples)

# Generate labels (is_fraud) with some randomness
is_fraud = np.random.choice([0, 1], n_samples, p=[0.95, 0.05])  # 5% fraud rate

# Create a DataFrame
data = pd.DataFrame({
    'transaction_amount': transaction_amount,
    'location': location,
    'customer_id': customer_id,
    'transaction_time': transaction_time,
    'device_type': device_type,
    'is_fraud': is_fraud
})

# Save the dataset to a CSV file
data.to_csv('synthetic_data.csv', index=False)
print('Synthetic data generated and saved to synthetic_data.csv')
