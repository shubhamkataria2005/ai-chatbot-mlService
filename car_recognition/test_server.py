import requests
import sys

def test_health():
    """Test the health endpoint"""
    try:
        response = requests.get('http://localhost:5002/health')
        print(f"Health check: {response.status_code}")
        print(f"Response: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Health check failed: {e}")
        return False

def test_brands():
    """Test the brands endpoint"""
    try:
        response = requests.get('http://localhost:5002/brands')
        print(f"Brands: {response.json()}")
        return response.status_code == 200
    except Exception as e:
        print(f"Brands check failed: {e}")
        return False

def test_predict(image_path):
    """Test the predict endpoint with an image"""
    try:
        with open(image_path, 'rb') as f:
            files = {'image': f}
            response = requests.post('http://localhost:5002/predict', files=files)
            print(f"Prediction result: {response.json()}")
            return response.status_code == 200
    except Exception as e:
        print(f"Prediction failed: {e}")
        return False

if __name__ == '__main__':
    print("Testing Car Recognition Server...")
    print("=" * 50)
    
    if test_health() and test_brands():
        print("\n✅ Server is running correctly!")
        
        if len(sys.argv) > 1:
            print(f"\nTesting with image: {sys.argv[1]}")
            test_predict(sys.argv[1])
    else:
        print("\n❌ Server is not running properly")