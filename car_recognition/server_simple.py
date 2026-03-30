# server_simple.py - Simple server that returns mock predictions
from flask import Flask, request, jsonify
from flask_cors import CORS
import random

app = Flask(__name__)
CORS(app)

# List of car brands
CAR_BRANDS = ['BMW', 'Mercedes', 'Audi', 'Toyota', 'Honda', 'Ford']

def get_mock_prediction():
    """Return a random car brand with mock confidence"""
    brands = CAR_BRANDS.copy()
    random.shuffle(brands)
    
    return {
        'success': True,
        'predicted_brand': brands[0],
        'confidence': round(random.uniform(70, 95), 1),
        'top_3': [
            {'brand': brands[0], 'confidence': round(random.uniform(70, 95), 1)},
            {'brand': brands[1], 'confidence': round(random.uniform(10, 30), 1)},
            {'brand': brands[2], 'confidence': round(random.uniform(5, 15), 1)}
        ],
        'model': 'mock_v1'
    }

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read the image (just to simulate processing)
        image_bytes = file.read()
        print(f"Received image: {len(image_bytes)} bytes")
        
        result = get_mock_prediction()
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    return jsonify({
        'status': 'ok',
        'model_loaded': True,
        'brands': CAR_BRANDS,
        'mode': 'mock'
    })

if __name__ == '__main__':
    port = 5002
    print("=" * 60)
    print("Car Recognition Server (Mock Mode)")
    print("=" * 60)
    print(f"Starting server on port {port}")
    print(f"Available brands: {CAR_BRANDS}")
    print("This server returns random predictions for testing")
    print("Server is ready! Waiting for requests...")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=True)