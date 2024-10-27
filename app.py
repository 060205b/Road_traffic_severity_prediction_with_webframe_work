from flask import Flask, request, render_template 
import numpy as np
import pickle
import os
import pandas as pd

app = Flask(__name__)

# Load the model
with open(r'models_pickle_file\traffic_accident_severity_model_new1.pkl', 'rb') as file:
    loaded_model = pickle.load(file)

# Mapping for categorical variables
age_band_mapping = {
    "1": 0,  # 18-30
    "2": 1,  # 31-50
    "3": 2,  # 51-65
    "4": 3   # 66+
}

experience_mapping = {
    "1-2yr": 0,
    "2-5yr": 1,
    "5-10yr": 2,
    "Above 10yr": 3,
    "Below 1yr": 4,
    "No Licence": 5,
    "Unknown": 6
}

weather_mapping = {
    "Normal": 0,
    "Raining": 1,
    "Raining and Windy": 2,
    "Cloudy": 3,
    "Other": 4,
    "Windy": 5,
    "Snow": 6,
    "Unknown": 7,
    "Fog or mist": 8
}

collision_mapping = {
    "Vehicle with vehicle collision": 0,
    "Collision with roadside-parked vehicles": 1,
    "Collision with roadside objects": 2,
    "Collision with animals": 3,
    "Other": 4,
    "Rollover": 5,
    "Fall from vehicles": 6,
    "Collision with pedestrians": 7,
    "With Train": 8,
    "Unknown": 9
}

cause_mapping = {
    "Moving Backward": 0,
    "Overtaking": 1,
    "Changing lane to the left": 2,
    "Changing lane to the right": 3,
    "Overloading": 4,
    "No priority to vehicle": 5,
    "No priority to pedestrian": 6,
    "Improper parking": 7,
    "Overspeed": 8,
    "Unknown": 9
}

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get data from the form
    age_band = request.form['ageRange']
    driving_experience = request.form['experience']
    weather_conditions = request.form['weather']
    type_of_collision = request.form['collision']
    cause_of_accident = request.form['cause']
    
    # Convert user inputs to numerical values
    try:
        input_data = pd.DataFrame({
            'Age_band_of_driver': [age_band_mapping[age_band]],
            'Driving_experience': [experience_mapping[driving_experience]],
            'Weather_conditions': [weather_mapping[weather_conditions]],
            'Type_of_collision': [collision_mapping[type_of_collision]],
            'Cause_of_accident': [cause_mapping[cause_of_accident]]
        })
    except KeyError as e:
        return render_template('result.html', prediction="Error", suggestion=f"Invalid input: {e}")

    # Make predictions
    prediction = loaded_model.predict(input_data)

    # Map numerical predictions to categories
    severity_mapping = {
        0: "Low",
        1: "Medium",
        2: "High"
    }

    predicted_severity = severity_mapping[prediction[0]]
    
    # Generate suggestions based on predicted severity
    if predicted_severity == "High":
        suggestion = "Ensure to take all necessary precautions when driving. Avoid busy roads if possible."
    elif predicted_severity == "Medium":
        suggestion = "Be aware of your surroundings and drive cautiously. Consider taking alternative routes."
    else:  # Low
        suggestion = "Safe driving conditions are predicted. Enjoy your drive!"

    return render_template('result.html', prediction=predicted_severity, suggestion=suggestion)

if __name__ == '__main__':
    app.run(debug=True)
