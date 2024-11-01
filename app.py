from fastapi import FastAPI, HTTPException
from fastapi.responses import JSONResponse
import pandas as pd
from pydantic import BaseModel
from src.pipelines.Waterlevel_pipeline import WaterLevelModel

# Define the request model for input JSON
class PredictionRequest(BaseModel):
    forward: int

app = FastAPI()

@app.get("/predict")
def predict(request: PredictionRequest):
    try:
        # Extract the 'forward' value from the request
        forward_steps = request.forward
        print(forward_steps)
        
        # Initialize your pipeline with station_code (assumed to be "PPB")
        pipeline = WaterLevelModel(station_code="PPB")
        
        # Run the prediction pipeline with 'forward_steps' (if applicable)
        prediction = pipeline.run_prediction_pipeline(forward_steps)
        prediction['timestamp'] = prediction['timestamp'].apply(lambda x: x.isoformat() if isinstance(x, pd.Timestamp) else x)
        prediction = prediction.to_dict(orient="records")
        
        # Prepare the response
        response = {
            "status": "Success",
            "data": prediction
        }
       
        return JSONResponse(content=response, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))