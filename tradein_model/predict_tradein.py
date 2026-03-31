
import sys
import json
import joblib
import pandas as pd
import numpy as np

def predict_tradein():
    try:
        # Load model and encoders
        model = joblib.load('tradein_model.pkl')
        scaler = joblib.load('scaler.pkl')
        features = joblib.load('features.pkl')
        numerical_cols = joblib.load('numerical_cols.pkl')
        
        label_encoders = {}
        categorical_cols = ['make', 'model', 'condition', 'body_type', 'fuel_type', 'transmission']
        for col in categorical_cols:
            label_encoders[col] = joblib.load(f'encoder_{col}.pkl')
        
        # Get input from command line
        input_data = json.loads(sys.argv[1])
        
        # Calculate age
        current_year = 2025
        age = current_year - input_data['year']
        
        # Condition score
        condition_scores = {'Excellent': 5, 'Very Good': 4, 'Good': 3, 'Fair': 2, 'Poor': 1}
        condition_score = condition_scores.get(input_data['condition'], 3)
        
        # Create input DataFrame
        input_dict = {
            'make': [input_data['make']],
            'model': [input_data['model']],
            'year': [input_data['year']],
            'mileage': [input_data['mileage']],
            'condition': [input_data['condition']],
            'body_type': [input_data['body_type']],
            'fuel_type': [input_data['fuel_type']],
            'transmission': [input_data['transmission']],
            'owners': [input_data['owners']],
            'engine_size': [input_data['engine_size']],
            'age': [age],
            'condition_score': [condition_score],
            'service_score': [2]
        }
        
        input_df = pd.DataFrame(input_dict)
        
        # Encode categorical variables
        for col in categorical_cols:
            if col in input_df.columns and col in label_encoders:
                input_df[col] = label_encoders[col].transform(input_df[col])
        
        # Scale numerical features
        for col in numerical_cols:
            if col in input_df.columns:
                values = input_df[col].values.reshape(-1, 1)
                input_df[col] = scaler.transform(values)
        
        # Ensure correct column order
        input_df = input_df[features]
        
        # Predict
        prediction = model.predict(input_df)[0]
        prediction = round(prediction / 100) * 100
        
        result = {
            'success': True,
            'predicted_value': float(prediction),
            'message': 'Prediction successful'
        }
        
        print(json.dumps(result))
        
    except Exception as e:
        error_result = {
            'success': False,
            'error': str(e),
            'message': 'Prediction failed'
        }
        print(json.dumps(error_result))

if __name__ == '__main__':
    predict_tradein()
