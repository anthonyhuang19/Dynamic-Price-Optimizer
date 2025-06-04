from fastapi import FastAPI, Form, HTTPException, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from fastapi.staticfiles import StaticFiles 
import src.serving.predict
import src.serving.monitor  # Import the monitoring module
import time
import numpy as np

app = FastAPI()

# Set up Jinja2 templates
templates = Jinja2Templates(directory="templates")

# Serve static files (CSS, JS, images, etc.)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Route to serve the index.html file directly
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

# Route to serve styles.css file directly
@app.get("/styles.css", response_class=FileResponse)
async def serve_styles():
    return FileResponse("static/styles.css")  # Assuming the CSS file is inside the "static" folder

# Input data model for prediction
class InputData(BaseModel):
    Number_of_Riders: float
    Number_of_Drivers: float
    interpolated_division: float
    Location_Category: str
    Vehicle_Type: str
    Time_of_Booking: float
    Expected_Ride_Duration: float

@app.post("/predict/", response_class=HTMLResponse)
async def predict_data(request: Request, 
                        Number_of_Riders: float = Form(...), 
                        Number_of_Drivers: float = Form(...), 
                        Location_Category: str = Form(...), 
                        Vehicle_Type: str = Form(...), 
                        Time_of_Booking: str = Form(...), 
                        Expected_Ride_Duration: float = Form(...)):
    
    try:
        # Preprocess the input features for prediction

        input_features = [
            Number_of_Riders,
            Number_of_Drivers,
            Time_of_Booking,
            Expected_Ride_Duration,
            Location_Category,
            Vehicle_Type,
        ]
        
        # Monitor and log request and prediction
        start_time = time.time()  # Track the start time to measure latency
        prediction,input = src.serving.predict.get_prediction(input_features)  # Call prediction function
        
        # Log prediction and request metrics
        latency = time.time() - start_time  # Calculate latency
        src.serving.monitor.log_request_metrics(input, prediction, latency)  # Log prediction
        src.serving.monitor.REQUEST_COUNT.inc()  # Increment the request count metric
        
        # Return the result and include the request for rendering the template
        return templates.TemplateResponse("index.html", {
            "request": request, 
            "prediction": round(prediction, 2), 
        })
    
    except Exception as e:
        # Log errors in case of a failure
        src.serving.monitor.ERROR_COUNT.inc()  # Increment error count metric
        src.serving.monitor.log_error(str(e))  # Log error to monitoring log
        
        # Raise HTTP Exception for the user
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

