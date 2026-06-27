import requests
import json
import base64
import time

URL_LOAD = "http://localhost:8082/models"
URL_TRANSFORM = "http://localhost:8082/models/fraud_model/transform"

def load_model():
    print("Loading model into MLeap runtime...")
    
    try:
        check_response = requests.get(f"{URL_LOAD}/fraud_model")
        if check_response.status_code == 200:
            print("Model is already loaded.")
            return True
    except Exception as e:
        print(f"Error checking model status: {e}")

    payload = {
        "modelName": "fraud_model",
        "uri": "file:/models/model.zip",
        "config": {
            "memoryTimeout": 900000,
            "diskTimeout": 900000
        },
        "force": False
    }
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(URL_LOAD, json=payload, headers=headers)
    
    if response.status_code in [200, 201, 202]:
        print("Model loaded successfully!")
        if response.status_code == 202:
            time.sleep(2)  # Wait for asynchronous load to finish
        return True
    else:
        print(f"Failed to load model: {response.status_code}")
        print(response.text)
        return False

def test_prediction():
    # Creating a sample transaction payload (High amount, Electronics, Late night)
    # The schema must match the features expected by the pipeline.
    # We provided amount, time_of_day, merchant_category, device_type in the pipeline assembler
    frame_json = {
      "schema": {
        "fields": [
          {"name": "amount", "type": "double"},
          {"name": "time_of_day", "type": "double"},
          {"name": "merchant_category", "type": "string"},
          {"name": "device_type", "type": "string"}
        ]
      },
      "rows": [
        [36000.0, 3.0, "Electronics", "Desktop"]
      ]
    }
    
    headers = {"Content-Type": "application/json"}
    
    print("\nSending transaction for scoring...")
    start_time = time.time()
    response = requests.post(URL_TRANSFORM, json=frame_json, headers=headers)
    latency = (time.time() - start_time) * 1000
    
    if response.status_code == 200:
        print(f"Prediction successful in {latency:.2f} ms")
        result_frame = response.json()
        
        # The result frame will have the original columns plus 'prediction' and 'probability'
        schema_fields = [f["name"] for f in result_frame["schema"]["fields"]]
        pred_idx = schema_fields.index("prediction")
        prob_idx = schema_fields.index("probability")
        
        row = result_frame["rows"][0]
        prediction = row[pred_idx]
        probability = row[prob_idx]
        
        print(f"Is Fraud? {'YES' if prediction == 1.0 else 'NO'}")
        if isinstance(probability, dict) and "values" in probability:
            print(f"Fraud Probability Score: {probability['values'][1]:.4f}")
        else:
            print(f"Fraud Probability Score: {probability}")
    else:
        print(f"Prediction failed: {response.status_code}")
        print(response.text)

if __name__ == "__main__":
    if load_model():
        test_prediction()
