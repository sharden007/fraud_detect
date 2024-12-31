from flask import Flask, request, jsonify
import pandas as pd
from tensorflow.keras.models import load_model

# Load the trained model
model = load_model('fraud_detection_model.h5')

# Load column names from training data for preprocessing consistency
data = pd.read_csv('synthetic_data.csv')
encoded_data = pd.get_dummies(data, columns=['device_type'], drop_first=True)
columns = encoded_data.drop('is_fraud', axis=1).columns

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Parse JSON input from request
    input_data = request.get_json()
    input_df = pd.DataFrame([input_data])

    # Ensure preprocessing matches training (e.g., one-hot encoding)
    input_encoded = pd.get_dummies(input_df)
    input_encoded = input_encoded.reindex(columns=columns, fill_value=0)

    # Make prediction using the trained model
    prediction = model.predict(input_encoded)[0][0]
    
    return jsonify({'fraud_probability': float(prediction)})

if __name__ == '__main__':
    app.run(debug=True)
