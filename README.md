# End-to-End Fraud Detection with PySpark & MLeap Serving

This project implements a complete, production-ready machine learning pipeline for real-time transaction fraud detection. It covers data generation, model training and serialization, containerized deployment, and client scoring.

## System Architecture

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
  (Generates csv)      ---------->         (Trains PySpark &)
                                          (exports model.zip)
                                                |
                                                v
[ test_prediction.py ]   <============>  [ REST API (8082) ]
 (Client API Scorer)      (JSON Payload)
```

---

## Prerequisites

- **Python 3.8 - 3.11** (PySpark 3.3.2 compatibility)
- **Java Development Kit (JDK) 8** (Installed and configured)
- **Docker Desktop** (To run the model serving container)

---

## Getting Started

### 1. Set Up Environment & Dependencies
Create a virtual environment and install the required dependencies:
```bash
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Generate Synthetic Data
Run the generator script to create a synthetic transaction dataset matching realistic fraud behaviors (high amounts, specific merchants, late-night hours):
```bash
python generate_data.py
```
This generates `transactions.csv` containing 10,000 records.

### 3. Build & Start the Docker Services
Start both the MLeap serving engine and the FastAPI gateway server simultaneously with Docker Compose:
```bash
docker-compose up --build -d
```
This builds the FastAPI image locally, starts both containers, mounts the local directory for database persistence, and launches the services.
- **FastAPI Gateway**: `http://localhost:8000`
- **MLeap Serving Engine**: `http://localhost:8082`

### 4. Train the Model and Export MLeap Bundle
Train the Spark ML pipeline using a Random Forest Classifier and export the trained model directly into an MLeap bundle (`model.zip`):
```bash
python train_model.py
```

### 5. Interactive Testing via Swagger UI
Open your browser and navigate to the FastAPI Swagger documentation:
```text
http://localhost:8000/docs
```
You can use the interactive interface to:
1. **Load the Model**: Call `POST /load-model` to load the exported `model.zip` into the MLeap runtime.
2. **Verify Status**: Call `GET /model-status` to check if the model is active.
3. **Score Transactions**: Call `POST /predict` with transaction details to run fraud prediction. Responses are automatically logged to a local SQLite database (`transactions.db`).
4. **Query Logs**: Call `GET /history` to fetch the logged history of scored transactions, prediction results, timestamps, and model latency metrics.

---

## Project Structure

- `app.py`: FastAPI gateway server with database logger.
- `Dockerfile`: Container definition for the FastAPI application.
- `docker-compose.yml`: Multi-container orchestration config defining MLeap and FastAPI.
- `generate_data.py`: Script to generate synthetic transactional records.
- `train_model.py`: PySpark training pipeline that exports the model to an MLeap bundle.
- `test_prediction.py`: Client script containing API calls to register and score transactions.
- `requirements.txt`: Python package dependencies.
- `hadoop/bin/`: Locally bundled Winutils helper binaries to run Spark natively on Windows.
