from fastapi import FastAPI, HTTPException
from database import InMemoryDatabase
import uvicorn
import pickle
import os
from pydantic import BaseModel
import numpy as np

import findspark

findspark.init('/opt/spark')

# Import Spark modules
from pyspark.sql import SparkSession
from pyspark.ml.regression import LinearRegressionModel
from pyspark.ml.feature import VectorAssembler

# Initialize Spark session
spark = SparkSession.builder \
    .appName("FlightDelayPredictionAPI") \
    .getOrCreate()
#    .config("spark.jars", "/path/to/spark-jar-files") \

app = FastAPI()

# Store the model and prediction history in memory
lr_model = None
prediction_history = []

# Define the input structure for predictions
class FlightData(BaseModel):
    dep_delay: float
    origin_wind_speed: float
    dest_wind_speed: float
    distance: float
    carrier_index: float

# Purpose:

# Health checks are commonly used in web services and applications to determine if the application is running properly and can respond to requests.
# This specific endpoint returns a simple response ({"status": "ok"}) when the service is healthy, meaning the server is up and running.

# How to use it:

# You or an automated system (such as a load balancer, monitoring service, or orchestration platform) can send a GET request to /health to verify that the API is operational.

@app.get("/health", status_code=200, tags=["health"], summary="Health check")
async def health():
    return {"status": "ok"}

# Load the model properly in /model/load/ route
@app.post("/model/load/")
async def load_model():
    global lr_model
    model_path = "/home/dbmello/Documents/case-machine-learning-engineer-pleno/src/model/lr_model_pyspark"
    
    lr_model = LinearRegressionModel.load(model_path)
    
    return {"status": "Model loaded successfully"}

# Correct predict function
@app.post("/model/predict/")
async def predict(data: dict):
    if lr_model is None:
        raise HTTPException(status_code=400, detail="Model is not loaded")
    
    # Prepare input features for prediction
    input_data = [
        data['dep_delay'],
        data['origin_wind_speed'],
        data['dest_wind_speed'],
        data['distance'],
        data['carrier_index']
    ]

    # Create a DataFrame with the input data
    input_df = spark.createDataFrame([input_data], ['dep_delay', 'origin_wind_speed', 'dest_wind_speed', 'distance', 'carrier_index'])
    
    # Use VectorAssembler to create a 'features' column
    assembler = VectorAssembler(inputCols=['dep_delay', 'origin_wind_speed', 'dest_wind_speed', 'distance', 'carrier_index'], outputCol="features")
    input_df = assembler.transform(input_df)

    # Make the prediction using the loaded PySpark model
    prediction = lr_model.transform(input_df)
    
    # Extract prediction result
    predicted_value = prediction.collect()[0]['prediction']

    # Store the input data and prediction in the history
    prediction_history.append({
        "input": data,
        "prediction": predicted_value
    })

    return {"prediction": predicted_value}

@app.get("/model/history/", tags=["model"], summary="Prediction history")
async def get_history():
    return {"history": prediction_history}

# How to use it:
# You can use these operations to manage user data, storing and retrieving information through API calls in your FastAPI application. Here’s how you could use each:

# To insert users, send a POST request with user information.
# To get user information, send a GET request with the user's name.
# To list all users, send a GET request to list the stored users.
# If you're building an application where you need to handle user information, these APIs provide a foundation for user-related operations. However, currently, the database is in-memory, meaning any stored data will be lost once the application is restarted. For persistent data storage, you'd replace InMemoryDatabase with an actual database, like MongoDB or PostgreSQL.

# User operations (remains unchanged)
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
    return {"status": "ok", "user": user}

@app.get("/user/", tags=["example"], summary="List all users")
async def list():
    db = InMemoryDatabase()
    users = db.get_collection('users')
    return {"status": "ok", "users": [x for x in users.find({}, {"_id": 0})]}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="debug")