# End-to-End Fraud Detection with PySpark & MLeap Serving

[![GitHub Repository](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/vikas9013/spark-mleap-fraud-detection)

This repository implements a production-ready, low-latency machine learning pipeline for real-time transaction fraud detection. It decouples high-throughput PySpark model training from real-time serving using MLeap serialization, running inside a containerized environment with FastAPI and SQLite persistence.

---

## 🏗️ System Architecture

```
                                      [ Docker Container ]
                                     +---------------------+
                                     |  MLeap Spring Boot  |
                                     |    Serving Engine   |
                                     +----------+----------+
                                                 ^
                                    (Loads Model | /models/model.zip)
                                                 |
[ generate_data.py ]                      [ train_model.py ]
  (Generates CSV)      ---------->         (Trains PySpark &)
                                          (exports model.zip)
                                                 |
                                                 v
[ test_prediction.py ]   <============>  [ REST API (8082) ]
  (Client API Scorer)      (JSON Payload)
```

---

## 📂 Project Structure

```text
.
├── hadoop/                   # Windows Hadoop binaries
│   └── bin/
│       ├── hadoop.dll
│       └── winutils.exe
├── app.py                    # FastAPI Gateway Server
├── Dockerfile                # Dockerfile for FastAPI Gateway
├── docker-compose.yml        # Multi-Container Orchestration (Gateway + MLeap)
├── generate_data.py          # Synthetic Transaction Data Generator
├── train_model.py            # PySpark ML training & MLeap serialization
├── test_prediction.py        # Client validation/scoring script
├── requirements.txt          # Python dependencies
├── transactions.csv          # Generated dataset (Git ignored)
├── model.zip                 # Serialized MLeap ML Bundle (Git ignored)
└── transactions.db           # SQLite log database (Git ignored)
```

---

## 🌟 Key Features

* **Decoupled Architecture**: Separation of heavy batch/offline training (PySpark) from fast, low-latency inference (MLeap C++ execution engine wrapped in Spring Boot).
* **Low-Latency Inference**: Achieves sub-15ms scoring latency by bypassing JVM/Python overhead during model prediction.
* **REST API Gateway**: Powered by **FastAPI** with auto-generated Swagger UI, request validation via **Pydantic**, and dependency injection.
* **Audit Logging & Analytics**: Automatic persistence of transactions, predictions, fraud probabilities, and latency metrics in a local **SQLite** database using **SQLAlchemy ORM**.
* **Windows Compatibility Patches**: Built-in configurations for Windows local execution (short-path handling for `JAVA_HOME` to resolve spacing issues, and pre-packaged Hadoop winutils).
* **Multi-Container Orchestration**: Fully containerized using **Docker Compose** with volume mounts for DB persistence and model loading.

---

## 🛠️ Tech Stack

* **Machine Learning**: PySpark 3.3.2, MLeap 0.24.0 (Spark & Bundle serialization)
* **API Gateway**: FastAPI, Uvicorn, Pydantic, Requests
* **Database & ORM**: SQLite, SQLAlchemy
* **Infrastructure**: Docker, Docker Compose, Windows Winutils Helpers
* **Testing**: Python Requests

---

## 🚀 Getting Started

### 1. Clone the Repository & Set Up Environment
Clone the repository to your local machine:
```bash
git clone https://github.com/vikas9013/spark-mleap-fraud-detection.git
cd spark-mleap-fraud-detection
```

Create a virtual environment and install the required dependencies:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Synthetic Transaction Data
Run the data generator to create a synthetic transaction dataset matching realistic fraud behaviors (high amounts, specific merchants, late-night transactions):
```bash
python generate_data.py
```
This generates `transactions.csv` containing 10,000 records.

### 3. Build & Start the Docker Services
Start both the MLeap serving engine and the FastAPI gateway server simultaneously with Docker Compose:
```bash
docker-compose up --build -d
```
This launches:
* **FastAPI Gateway**: `http://localhost:8000`
* **MLeap Serving Engine**: `http://localhost:8082`

### 4. Train the Model and Export the MLeap Bundle
Train the Spark ML pipeline using a Random Forest Classifier and serialize it directly into an MLeap bundle (`model.zip`):
```bash
python train_model.py
```

### 5. Interactive Testing via Swagger UI
Open your browser and navigate to:
```text
http://localhost:8000/docs
```
Use the interactive Swagger documentation to test the endpoints:
1. **Load the Model**: Call `POST /load-model` to load the exported `model.zip` into the MLeap runtime.
2. **Verify Status**: Call `GET /model-status` to check if the model is active.
3. **Score Transactions**: Call `POST /predict` with transaction details to run fraud prediction. Responses are automatically logged to the SQLite database.
4. **Query Logs**: Call `GET /history` to fetch the logged history of scored transactions, prediction results, timestamps, and model latency metrics.

---

## 📡 API Reference

### 1. Check Model Status
* **Endpoint**: `GET /model-status`
* **Response**:
  ```json
  {
    "status": "loaded",
    "model_details": {
      "name": "fraud_model",
      "uri": "file:/models/model.zip"
    }
  }
  ```

### 2. Load Model
* **Endpoint**: `POST /load-model`
* **Parameters**: `force` (bool)
* **Response**:
  ```json
  {
    "status": "success",
    "message": "Model loaded successfully."
  }
  ```

### 3. Score Transaction
* **Endpoint**: `POST /predict`
* **Request Payload**:
  ```json
  {
    "amount": 35000.0,
    "time_of_day": 3.0,
    "merchant_category": "Electronics",
    "device_type": "Desktop"
  }
  ```
* **Response**:
  ```json
  {
    "is_fraud": true,
    "fraud_probability": 0.548,
    "latency_ms": 10.24
  }
  ```

### 4. Fetch Scoring Logs
* **Endpoint**: `GET /history?limit=100`
* **Response**: List of logged transactions containing `id`, `amount`, `time_of_day`, `merchant_category`, `device_type`, `is_fraud`, `fraud_probability`, `latency_ms`, and `timestamp`.

---

## 🗄️ Database Schema & Feature Engineering Details

### 1. Database Table (`transaction_logs`)
The SQLite database stores logs of scored transactions. The schema is mapped using SQLAlchemy ORM:
* **`id`** (INTEGER, Primary Key): Unique auto-incrementing ID.
* **`amount`** (FLOAT): The transaction amount in INR.
* **`time_of_day`** (FLOAT): The hour of the transaction (0-23).
* **`merchant_category`** (VARCHAR): The merchant category class.
* **`device_type`** (VARCHAR): The device class used.
* **`is_fraud`** (BOOLEAN): Classification output from the model (`true` if fraud, else `false`).
* **`fraud_probability`** (FLOAT): The Random Forest model confidence score for fraud.
* **`latency_ms`** (FLOAT): The end-to-end model transformation execution time in milliseconds.
* **`timestamp`** (DATETIME): Auto-generated timestamp of the request.

### 2. Feature Classes
The categorical fields configured during `train_model.py` and handled by the MLeap pipeline support the following values:
* **Merchant Categories**: `Groceries`, `Electronics`, `Clothing`, `Restaurants`, `Travel`, `OnlineRetail`.
* **Device Types**: `Mobile`, `Desktop`, `Tablet`.

---

## 📜 License

This project is open-source and available under the [MIT License](https://opensource.org/licenses/MIT).
