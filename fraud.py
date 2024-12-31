import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

# Step 1: Simulate a small dataset (or load from a CSV file)
data = pd.DataFrame({
    'transaction_amount': [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000],
    'location': [1, 2, 1, 2, 1, 2, 1, 2, 1, 2],  # Encoded locations (e.g., city IDs)
    'customer_id': [101, 102, 103, 104, 105, 106, 107, 108, 109, 110],
    'is_fraud': [0, 1, 0, 1, 0, 1, 0, 1, 0, 1]   # Labels: Fraud (1) or Not Fraud (0)
})

# Step 2: Split dataset into features (X) and target labels (y)
X = data.drop("is_fraud", axis=1)
y = data["is_fraud"]

# Step 3: Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Build and train the Random Forest Classifier
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Step 5: Make predictions on the test set
predictions = model.predict(X_test)

# Step 6: Evaluate the model using classification metrics
print("Classification Report:")
print(classification_report(y_test, predictions))

print("Confusion Matrix:")
print(confusion_matrix(y_test, predictions))
