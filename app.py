from fastapi import FastAPI, HTTPException, status, Depends
from pydantic import BaseModel, Field
import requests
import time
import os
from datetime import datetime
from sqlalchemy import create_engine, Column, Integer, Float, String, Boolean, DateTime
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session

# FastAPI App Configuration
app = FastAPI(
    title="Fraud Detection API Gateway",
    description="A Swagger-equipped FastAPI gateway wrapping the MLeap serving microservice for transaction fraud scoring, integrated with SQLite database persistence.",
    version="1.1.0"
)

# Read environment variables
MLEAP_SERVER_URL = os.getenv("MLEAP_SERVER_URL", "http://localhost:8082")
MODEL_NAME = "fraud_model"

# Database Configuration (SQLite)
DATABASE_URL = "sqlite:///./transactions.db"
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()

# SQLAlchemy Database Model
class TransactionLog(Base):
    __tablename__ = "transaction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    amount = Column(Float, nullable=False)
    time_of_day = Column(Float, nullable=False)
    merchant_category = Column(String, nullable=False)
    device_type = Column(String, nullable=False)
    is_fraud = Column(Boolean, nullable=False)
    fraud_probability = Column(Float, nullable=False)
    latency_ms = Column(Float, nullable=False)
    timestamp = Column(DateTime, default=datetime.utcnow)

# Create Database tables
Base.metadata.create_all(bind=engine)

# DB Session Dependency
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# Pydantic Schemas
class TransactionRequest(BaseModel):
    amount: float = Field(..., description="The transaction amount in INR (Indian Rupee).", example=35000.0)
    time_of_day: float = Field(..., description="The hour of the transaction (0-23).", example=3.0)
    merchant_category: str = Field(..., description="Category of merchant (e.g. Groceries, Electronics, Restaurants).", example="Electronics")
    device_type: str = Field(..., description="Device used for transaction (Mobile, Desktop, Tablet).", example="Desktop")

class PredictionResponse(BaseModel):
    is_fraud: bool = Field(..., description="True if the model classifies this transaction as fraud, False otherwise.")
    fraud_probability: float = Field(..., description="Confidence score/probability of the transaction being fraud.")
    latency_ms: float = Field(..., description="Model transformation latency in milliseconds.")

class TransactionHistoryResponse(BaseModel):
    id: int
    amount: float
    time_of_day: float
    merchant_category: str
    device_type: str
    is_fraud: bool
    fraud_probability: float
    latency_ms: float
    timestamp: datetime

    class Config:
        from_attributes = True

# API Endpoints
@app.get("/model-status", summary="Check Model Status", description="Checks if the fraud detection model is currently loaded in the MLeap container.")
def get_model_status():
    try:
        response = requests.get(f"{MLEAP_SERVER_URL}/models/{MODEL_NAME}")
        if response.status_code == 200:
            return {
                "status": "loaded",
                "model_details": response.json()
            }
        else:
            return {"status": "not_loaded", "detail": f"MLeap server returned code {response.status_code}"}
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MLeap serving container is unreachable. Make sure Docker is running."
        )

@app.post("/load-model", summary="Load Model", description="Loads the exported ML model zip into the container runtime.")
def load_model(force: bool = False):
    try:
        check_response = requests.get(f"{MLEAP_SERVER_URL}/models/{MODEL_NAME}")
        if check_response.status_code == 200 and not force:
            return {"status": "success", "message": "Model was already loaded."}
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MLeap serving container is unreachable. Make sure Docker is running."
        )

    payload = {
        "modelName": MODEL_NAME,
        "uri": "file:/models/model.zip",
        "config": {
            "memoryTimeout": 900000,
            "diskTimeout": 900000
        },
        "force": force
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(f"{MLEAP_SERVER_URL}/models", json=payload, headers=headers)
        if response.status_code in [200, 201, 202]:
            if response.status_code == 202:
                time.sleep(2.0)  # Wait for asynchronous loading
            return {"status": "success", "message": "Model loaded successfully."}
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"Failed to load model into MLeap runtime: {response.text}"
            )
    except requests.exceptions.RequestException as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error communicating with MLeap: {str(e)}"
        )

@app.post("/predict", response_model=PredictionResponse, summary="Predict Fraud", description="Scores a transaction to evaluate if it is fraudulent, and logs the result to the SQLite database.")
def predict(request: TransactionRequest, db: Session = Depends(get_db)):
    # Construct MLeap LeapFrame structure
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
            [request.amount, request.time_of_day, request.merchant_category, request.device_type]
        ]
    }
    
    headers = {"Content-Type": "application/json"}
    start_time = time.time()
    
    try:
        response = requests.post(
            f"{MLEAP_SERVER_URL}/models/{MODEL_NAME}/transform",
            json=frame_json,
            headers=headers
        )
        latency = (time.time() - start_time) * 1000
        
        if response.status_code == 200:
            result_frame = response.json()
            
            schema_fields = [f["name"] for f in result_frame["schema"]["fields"]]
            pred_idx = schema_fields.index("prediction")
            prob_idx = schema_fields.index("probability")
            
            row = result_frame["rows"][0]
            prediction = row[pred_idx]
            probability_tensor = row[prob_idx]
            
            # Extract probability of class 1.0 (fraud)
            fraud_prob = probability_tensor["values"][1] if isinstance(probability_tensor, dict) else probability_tensor[1]
            
            # Log prediction to database
            log_entry = TransactionLog(
                amount=request.amount,
                time_of_day=request.time_of_day,
                merchant_category=request.merchant_category,
                device_type=request.device_type,
                is_fraud=bool(prediction == 1.0),
                fraud_probability=float(fraud_prob),
                latency_ms=round(latency, 2)
            )
            db.add(log_entry)
            db.commit()
            db.refresh(log_entry)
            
            return PredictionResponse(
                is_fraud=log_entry.is_fraud,
                fraud_probability=log_entry.fraud_probability,
                latency_ms=log_entry.latency_ms
            )
        elif response.status_code == 404:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Model not loaded. Please call /load-model endpoint first."
            )
        else:
            raise HTTPException(
                status_code=response.status_code,
                detail=f"MLeap prediction server error: {response.text}"
            )
            
    except requests.exceptions.ConnectionError:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="MLeap serving container is unreachable. Make sure Docker is running."
        )

@app.get("/history", response_model=list[TransactionHistoryResponse], summary="Get Prediction History", description="Queries all logged transactions and predictions from the SQLite database.")
def get_history(limit: int = 100, db: Session = Depends(get_db)):
    logs = db.query(TransactionLog).order_by(TransactionLog.timestamp.desc()).limit(limit).all()
    return logs
