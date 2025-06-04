import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

class PredictRideCost:
    
    @staticmethod
    def _map_location_category(series: pd.Series) -> pd.Series:
        Location_Mapping = {'Urban': 0, 'Suburban': 1, 'Rural': 2}
        return series.map(Location_Mapping)
    
    @staticmethod
    def _map_vehicle_type(series: pd.Series) -> pd.Series:
        Vehicle_Mapping = {'Premium': 0, 'Economy': 1}
        return series.map(Vehicle_Mapping)

    @staticmethod
    def _map_time_of_booking(series: pd.Series) -> pd.Series:
        time_mapping = {'Morning': 0, 'Afternoon': 1, 'Evening': 2, 'Night': 3}
        return series.map(time_mapping)
    
    # Function to load the model
    @staticmethod
    def load_model(model_path: str):
        return joblib.load(model_path)

    # Function to load the scaler
    @staticmethod
    def load_scaler(scaler_path: str):
        return joblib.load(scaler_path)

    # Function to structure the input data to match the model's expected features
    @staticmethod
    def structure_input_data(input_data: list) -> pd.DataFrame:
        # Correct feature names used during model training
        feature_names = [
            "Number_of_Riders", 
            "Number_of_Drivers", 
            "interpolated_division",
            "Location_Category",   # Location Category (one column, no dummies)
            "Vehicle_Type",        # Vehicle Type (one column, no dummies)
            "Time_of_Booking",     # Time of Booking
            "Expected_Ride_Duration"  # Ride duration
        ]
        
        # Convert the input data into a pandas DataFrame
        structured_data = pd.DataFrame([input_data], columns=feature_names)
        
        # Apply mappings to categorical variables
        structured_data["Location_Category"] = PredictRideCost._map_location_category(structured_data["Location_Category"])
        structured_data["Vehicle_Type"] = PredictRideCost._map_vehicle_type(structured_data["Vehicle_Type"])
        structured_data["Time_of_Booking"] = PredictRideCost._map_time_of_booking(structured_data["Time_of_Booking"])

        return structured_data

    # Function to make a prediction
    @staticmethod
    def predict(model, scaler, input_data: pd.DataFrame):
        # Ensure the input data is scaled correctly
        scaled_input_data = scaler.transform(input_data)
        
        # Make prediction using the model
        prediction = model.predict(scaled_input_data)
        
        return prediction

    # Main function to integrate everything
    @staticmethod
    def main(input_data: list):
        # Load the trained model and scaler
        model = PredictRideCost.load_model("model/model.pkl")
        scaler = PredictRideCost.load_scaler("model/scaler.pkl")
        
        # Structure the input data (must match the training data format)
        structured_input = PredictRideCost.structure_input_data(input_data)
        
        # Make the prediction
        prediction = PredictRideCost.predict(model, scaler, structured_input)
        
        # Print the result
        print("Prediction:", prediction)
        return prediction

# Example usage: providing the input data as a list
if __name__ == "__main__":
    input_data = [
        90,           # Number_of_Riders
        45,           # Number_of_Drivers
        0.914883,     # interpolated_division
        'Urban',      # Location_Category (string, one of: Urban, Suburban, Rural)
        'Premium',    # Vehicle_Type (string, one of: Premium, Economy)
        'Morning',    # Time_of_Booking (string, one of: Morning, Afternoon, Evening, Night)
        90            # Expected_Ride_Duration
    ]
    
    # Call the main function
    PredictRideCost.main(input_data)
