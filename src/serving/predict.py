import joblib
import numpy as np
from sklearn.preprocessing import LabelEncoder
import traceback
from src.model.predict import *



def get_prediction(features: list):
    try:
        print(f"Raw input prediction: {features}")
        poly = PredictRideCost.load_model("model/polynomial_model.pkl")
        
        # Assuming features = [riders, drivers, loc_cat_encoded, veh_type_encoded, ..., time, duration]
        #[90.0, 45.0, 0, 90.0, 0, 0]
        riders = features[0]
        print("hello1")
        drivers = features[1]
        print("hello2")
        location_category_encoded = features[4]
        print("hello3")
        vehicle_type_encoded = features[5]
        print("hello4")
        time_of_booking = features[2]
        print("hello5")
        ride_duration = features[3]
        print("hello6")

        # Compute interpolated division using saved poly
        interpolated_division = poly(riders / drivers)
        print("hello")

        # Construct final feature list in correct order
        final_features = [
            riders,
            drivers,
            interpolated_division,
            location_category_encoded,
            vehicle_type_encoded,
            time_of_booking,
            ride_duration
        ]

        print(f"Processed features: {final_features}")
        
        
        prediction = PredictRideCost.main(final_features)
        return float(prediction[0]),final_features
    except Exception as e:
        import traceback
        traceback.print_exc()
        raise Exception(f"Error during prediction: {str(e)}")

#uvicorn src.serving.app:app --reload