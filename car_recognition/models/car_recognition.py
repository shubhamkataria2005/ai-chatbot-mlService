import tensorflow as tf
import numpy as np
from PIL import Image
import json
import sys
import os
import traceback

def predict_car(image_path):
    try:
        print(f"Loading image from: {image_path}")

        # Check if image exists
        if not os.path.exists(image_path):
            return {"success": False, "error": f"Image file not found: {image_path}"}

        # Get script directory
        script_dir = os.path.dirname(os.path.abspath(__file__))
        model_path = os.path.join(script_dir, 'car_model.h5')

        print(f"Model path: {model_path}")
        print(f"Model exists: {os.path.exists(model_path)}")

        if not os.path.exists(model_path):
            return {"success": False, "error": f"Model file not found: {model_path}"}

        # Load model
        print("Loading TensorFlow model...")
        model = tf.keras.models.load_model(model_path)
        print("Model loaded successfully")

        # Define car brands (update this list based on your model's training)
        car_brands = ['BMW', 'Mercedes', 'Audi', 'Toyota', 'Honda', 'Ford']  # Example brands

        # Load and prepare image
        print("Processing image...")
        img = Image.open(image_path)
        img = img.resize((150, 150))  # Adjust based on your model's expected input
        img_array = np.array(img) / 255.0

        # Handle different image formats
        if len(img_array.shape) == 2:  # Grayscale
            img_array = np.stack([img_array] * 3, axis=-1)
        elif img_array.shape[2] == 4:  # RGBA
            img_array = img_array[:, :, :3]

        img_array = np.expand_dims(img_array, axis=0)

        print(f"Image shape: {img_array.shape}")

        # Make prediction
        print("Making prediction...")
        prediction = model.predict(img_array)[0]
        top_index = np.argmax(prediction)
        confidence = float(prediction[top_index]) * 100

        # Get top 3 predictions
        top_indices = np.argsort(prediction)[-3:][::-1]
        top_predictions = []

        for idx in top_indices:
            if idx < len(car_brands):
                top_predictions.append({
                    "brand": car_brands[idx],
                    "confidence": float(prediction[idx]) * 100
                })

        result = {
            "success": True,
            "predicted_brand": car_brands[top_index] if top_index < len(car_brands) else "Unknown",
            "confidence": confidence,
            "all_predictions": top_predictions,
            "model": "Car_Recognizer_v1.0",
            "image_size": f"{img.size[0]}x{img.size[1]}"
        }

        print(f"Prediction successful: {result['predicted_brand']} ({confidence:.2f}%)")
        return result

    except Exception as e:
        error_msg = f"Prediction failed: {str(e)}"
        print(f"ERROR: {error_msg}")
        traceback.print_exc()
        return {"success": False, "error": error_msg}

# Main execution
if __name__ == "__main__":
    try:
        print("Python car recognition script started")
        print(f"Arguments: {sys.argv}")

        # Read input from file
        if len(sys.argv) > 1:
            input_file = sys.argv[1]
            print(f"Reading input from: {input_file}")

            with open(input_file, 'r') as f:
                input_data = json.load(f)
        else:
            print("ERROR: No input file provided")
            sys.exit(1)

        image_path = input_data['image_path']
        print(f"Processing image: {image_path}")

        # Make prediction
        result = predict_car(image_path)

        # Output result as JSON
        print(json.dumps(result))

    except Exception as e:
        error_msg = f"Script execution failed: {str(e)}"
        print(f"ERROR: {error_msg}")
        traceback.print_exc()

        error_result = {
            "success": False,
            "error": error_msg
        }
        print(json.dumps(error_result))