import requests
import json

BASE_URL = "http://127.0.0.1:5000"

def check_health():
    url = f"{BASE_URL}/health"
    response = requests.get(url)
    print("=== Health Check ===")
    print(response.json(), "\n")

def single_prediction(text):
    url = f"{BASE_URL}/predict"
    payload = {"text": text}
    response = requests.post(url, json=payload)
    print("=== Single Prediction ===")
    print(json.dumps(response.json(), indent=2), "\n")

def batch_prediction(texts):
    url = f"{BASE_URL}/predict/batch"
    payload = {"texts": texts}
    response = requests.post(url, json=payload)
    print("=== Batch Prediction ===")
    print(json.dumps(response.json(), indent=2), "\n")

def get_stats():
    url = f"{BASE_URL}/stats"
    response = requests.get(url)
    print("=== API Stats ===")
    print(json.dumps(response.json(), indent=2), "\n")

if __name__ == "__main__":
    # Check API health
    check_health()

    # Test single prediction
    single_prediction("The product quality is fantastic and delivery was fast!")

    # Test batch prediction
    batch_prediction([
        "Worst customer service ever!",
        "I love this product, will buy again",
        "Average experience, nothing special"
    ])

    # Get API stats
    get_stats()
