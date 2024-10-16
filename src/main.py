from fastapi import FastAPI, HTTPException, Response
from fastapi.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
import json
from contextlib import asynccontextmanager
from database import InMemoryDatabase
import uvicorn
import pickle
import os
from pydantic import BaseModel
import numpy as np
import pandas as pd

# Import Spark modules
from pyspark.sql import SparkSession
from pyspark.ml import PipelineModel

# Initialize Spark session
spark = SparkSession.builder \
    .appName("FlightDelayPredictionAPI") \
    .getOrCreate()

# Set Spark logging level to reduce verbosity
spark.sparkContext.setLogLevel("ERROR")

@asynccontextmanager
async def lifespan(app: FastAPI):
    global model
    model_type = os.getenv('MODEL_TYPE', 'pyspark')
    model = None
    try:
        if model_type == 'pyspark':
            model_path = "./models/lr_model_pyspark"
            model = PipelineModel.load(model_path)
            print("PySpark PipelineModel loaded successfully.")
        elif model_type == 'pickle':
            model_path = "./models/lr_model.pkl"
            with open(model_path, 'rb') as file:
                model = pickle.load(file)
            print("Pickle model loaded successfully.")
        else:
            print(f"Unknown MODEL_TYPE: {model_type}")
    except Exception as e:
        print(f"Error loading model: {e}")
    
    yield
    print("Application is shutting down.")

    
app = FastAPI(lifespan=lifespan)

# Store the model and prediction history in memory
model = None
prediction_history = []

# Define the input structure for predictions
class FlightData(BaseModel):
    dep_delay: float
    origin_wind_speed: float
    dest_wind_speed: float
    distance: float
    carrier: str

@app.get("/health", status_code=200, tags=["health"], summary="Health check")
async def health():
    return {"status": "ok"}

@app.post("/model/predict/")
async def predict(data: FlightData):
        
    print(f"Received data: {data}")
    print(f"Parsed as: {data.dict()}")

    # Prepare input features for prediction
    input_data = {
        'dep_delay': data.dep_delay,
        'origin_wind_speed': data.origin_wind_speed,
        'dest_wind_speed': data.dest_wind_speed,
        'distance': data.distance,
        'carrier': data.carrier
    }

    model_type = os.getenv('MODEL_TYPE', 'pyspark')
    
    print(f"Imprimindo MODEL_TYPE: {model_type == 'pyspark'}")
    print(model)
    
    if model_type == 'pyspark':
        try:
            input_df = spark.createDataFrame([input_data])
            prediction = model.transform(input_df)
            predicted_value = prediction.collect()[0]['prediction']
        except Exception as e:
            print(f"PySpark prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    elif model_type == 'pickle':
        try:
            df_input = pd.DataFrame([input_data])
            predicted_value = model.predict(df_input)[0]
        except Exception as e:
            print(f"Pickle prediction error: {e}")
            raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")
    
    else:
        raise HTTPException(status_code=400, detail=f"Unknown model type: {model_type}")

    prediction_history.append({
        "input": input_data,
        "prediction": predicted_value
    })

    return {"prediction": predicted_value}

@app.get("/model/history/", tags=["model"], summary="Prediction history")
async def get_history():
    json_history = jsonable_encoder({"history": prediction_history})
    formatted_history = json.dumps(json_history, indent=2)
    return Response(content=formatted_history, media_type="application/json")

@app.post("/user/", tags=["example"], summary="Insert user")
async def insert(data: dict):
    db = InMemoryDatabase()
    users = db.get_collection('users')
    users.insert_one(data)
    return {"status": "ok"}

@app.get("/user/{name}", status_code=200, tags=["example"], summary="Get user by name")
async def get(name: str):
    db = InMemoryDatabase()
    users = db.get_collection('users')
    user = users.find_one({"name": name})
    json_user = jsonable_encoder({"status": "ok", "user": user})
    formatted_user = json.dumps(json_user, indent=4)
    return Response(content=formatted_user, media_type="application/json")

@app.get("/user/", tags=["example"], summary="List all users")
async def list():
    db = InMemoryDatabase()
    users = db.get_collection('users')
    json_users = jsonable_encoder({"status": "ok", "users": [x for x in users.find({}, {"_id": 0})]})
    formatted_users = json.dumps(json_users, indent=4)
    return Response(content=formatted_users, media_type="application/json")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")

