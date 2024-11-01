# **Road Traffic Accident Severity Prediction**

## **Overview**

This project is a Traffic Accident Severity Prediction System that utilizes machine learning to predict the severity of a traffic accident based on various input parameters, such as the age of the driver, weather conditions, type of collision, driving experience, and cause of the accident. The model is built using a Random Forest Classifier, tuned with GridSearchCV for optimal hyperparameters. The application is implemented in Flask, providing a user-friendly web interface where users can input parameters and receive predictions.

## **Modeling Process**

### **Data Preprocessing**

The dataset contains features such as:
- **Age_band_of_driver**
- **Driving_experience**
- **Weather_conditions**
- **Type_of_collision**
- **Cause_of_accident**

The data was split into training and test sets (X_train, X_test, y_train, y_test) to evaluate the model.

### **Model Selection**

A **Random Forest Classifier** was selected for the prediction task. The model was trained with class balancing techniques to handle imbalanced data effectively.

### **Hyperparameter Tuning**

**GridSearchCV** was employed to tune hyperparameters such as:
- **Number of estimators**
- **Max depth**
- **Minimum samples split**

This ensures optimal model performance.

### **Model Saving**

After training and tuning, the best model was saved using **Pickle** for future use.

## **Features**

- **Machine Learning Prediction**: A tuned Random Forest model is used to predict traffic accident severity (Low, Medium, High).
- **Web Interface**: Built with Flask, allowing users to interact with the model via a simple web form.
- **Feature Importances**: Visualized to show which features have the most impact on accident severity predictions.
- **SHAP and LIME Plots**: Integrated visualizations to explain model predictions, providing insights into how each feature affects the prediction outcomes.

##**Clone the Repository**

git clone https://github.com/yourusername/traffic-accident-severity-prediction.git
cd traffic-accident-severity-prediction

##**Set Up a Virtual Environment (optional but recommended)**
For macOS/Linux:
venv/bin/activate
For Windows:
venv\Scripts\activate

##**Run the Flask Application**
python app.py
The application will run on http://127.0.0.1:5000/. Open this link in your browser.

##**Dependencies**
Flask: To build the web application
scikit-learn: For machine learning model building and tuning
Pandas: For data manipulation
Matplotlib: For plotting feature importances
SHAP: For model interpretation and explanation
LIME: For model interpretation and explanation
Pickle: To save and load machine learning models

##**Video of Web Application**
Home Page (Prediction Form) Prediction Result

[Home Page (Prediction Form)
Prediction Result](https://github.com/060205b/Road_traffic_severity_prediction_with_webframe_work/blob/main/Prediction_video.mp4)

