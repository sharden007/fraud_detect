import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Load the synthetic data
data = pd.read_csv('synthetic_data.csv')

# One-hot encode the 'device_type' feature
encoded_data = pd.get_dummies(data, columns=['device_type'], drop_first=True)

# Separate features and target variable
X = encoded_data.drop('is_fraud', axis=1)
y = encoded_data['is_fraud']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build a simple neural network model
model = Sequential([
    Dense(32, input_dim=X_train.shape[1], activation='relu'),
    Dense(16, activation='relu'),
    Dense(1, activation='sigmoid')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)

# Save the trained model
model.save('fraud_detection_model.h5')
print('Model trained and saved to fraud_detection_model.h5')
