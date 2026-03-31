# server.py - Standalone ML server for car recognition
from flask import Flask, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import numpy as np
from PIL import Image
import io
import json
import os
import sys
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create the Flask app
app = Flask(__name__)
CORS(app)  # This allows your React frontend to talk to this server

# Get the directory where this script is located
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, 'models', 'car_model.h5')

print("=" * 60)
print("Car Recognition ML Server")
print("=" * 60)
print(f"Script directory: {SCRIPT_DIR}")
print(f"Model path: {MODEL_PATH}")
print(f"Model exists: {os.path.exists(MODEL_PATH)}")

# Load the model
print(f"Loading model from: {MODEL_PATH}")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print("✓ Model loaded successfully")
    
    # Print model summary
    model.summary()
    
except Exception as e:
    print(f"❌ Failed to load model: {e}")
    model = None

# The car brands your model can recognize (from your model's training)
# You should update this list based on your actual model's classes
CAR_BRANDS = ['BMW', 'Mercedes', 'Audi', 'Toyota', 'Honda', 'Ford', 'Volkswagen', 'Nissan', 'Hyundai', 'Kia']

def preprocess_image(image_bytes):
    """
    Prepare the image for the model
    This converts the uploaded image to the format the model expects
    """
    try:
        # Open the image from bytes
        image = Image.open(io.BytesIO(image_bytes))
        
        # Convert to RGB (in case it's PNG with transparency)
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Resize to 150x150 (this is what your model expects)
        image = image.resize((150, 150))
        
        # Convert to array of numbers (0-255)
        img_array = np.array(image)
        
        # Normalize: convert 0-255 to 0-1 (makes the model work better)
        img_array = img_array / 255.0
        
        # Add batch dimension (the model expects multiple images, we have 1)
        img_array = np.expand_dims(img_array, axis=0)
        
        logger.info(f"Image shape after preprocessing: {img_array.shape}")
        
        return img_array
        
    except Exception as e:
        logger.error(f"Error preprocessing image: {e}")
        raise

@app.route('/predict', methods=['POST'])
def predict():
    """
    Endpoint that receives an image and returns the predicted car brand
    """
    # Check if model is loaded
    if model is None:
        return jsonify({
            'success': False,
            'error': 'Model not loaded'
        }), 500
    
    try:
        # Get the image from the request
        if 'image' not in request.files:
            return jsonify({'success': False, 'error': 'No image provided'}), 400
        
        file = request.files['image']
        if file.filename == '':
            return jsonify({'success': False, 'error': 'No file selected'}), 400
        
        # Read image bytes
        image_bytes = file.read()
        logger.info(f"Received image: {len(image_bytes)} bytes")
        
        # Preprocess the image
        img_array = preprocess_image(image_bytes)
        
        # Make prediction
        predictions = model.predict(img_array)[0]
        
        # Get top 3 predictions
        top_indices = np.argsort(predictions)[-3:][::-1]
        top_3 = []
        for idx in top_indices:
            if idx < len(CAR_BRANDS):
                top_3.append({
                    'brand': CAR_BRANDS[idx],
                    'confidence': float(predictions[idx]) * 100
                })
        
        # Get the best prediction
        best_idx = np.argmax(predictions)
        best_brand = CAR_BRANDS[best_idx] if best_idx < len(CAR_BRANDS) else 'Unknown'
        best_confidence = float(predictions[best_idx]) * 100
        
        result = {
            'success': True,
            'predicted_brand': best_brand,
            'confidence': best_confidence,
            'top_3': top_3,
            'model': 'car_recognition_v1'
        }
        
        logger.info(f"Prediction result: {best_brand} ({best_confidence:.2f}%)")
        
        return jsonify(result)
        
    except Exception as e:
        logger.error(f"Error during prediction: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@app.route('/health', methods=['GET'])
def health():
    """
    Health check endpoint - lets your backend know the server is running
    """
    return jsonify({
        'status': 'ok',
        'model_loaded': model is not None,
        'brands': CAR_BRANDS,
        'server': 'car_recognition_v1'
    })

@app.route('/brands', methods=['GET'])
def get_brands():
    """
    Returns the list of car brands the model can recognize
    """
    return jsonify({
        'success': True,
        'brands': CAR_BRANDS,
        'count': len(CAR_BRANDS)
    })

# Run the server
if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5002))
    print(f"\nStarting car recognition server on port {port}")
    print(f"Available brands: {CAR_BRANDS}")
    print("Server is ready! Waiting for requests...")
    print(f"\nTest the server:")
    print(f"  Health check: http://localhost:{port}/health")
    print(f"  List brands: http://localhost:{port}/brands")
    print(f"  Predict endpoint: POST http://localhost:{port}/predict")
    print("=" * 60)
    app.run(host='0.0.0.0', port=port, debug=True)