# **Heart Disease Prediction Web App**

This is a simple Flask web application for predicting the likelihood of heart disease based on user input. The application uses a pre-trained machine learning model that combines Support Vector Machine (SVM) and Random Forest classifiers. The model was trained on a dataset available in 'new_g40.csv'.

## **Installation**

1. Clone the repository:
git clone -  https://github.com/shubhankardutta38/Heart_Disease_Prediction_Using_Python.git

2. Navigate to the project directory:
cd heart-disease-prediction

3. Install the required dependencies

## Usage

1. Ensure that you have Python and Flask installed on your system.
2. Run the Flask application:
   python main.py
3. Open your web browser and go to http://127.0.0.1:5000
4. Fill out the form with the required information, and click the "Predict" button to get the prediction result.

## Project Structure

main.py: The main Flask application file containing the web server logic.
templates/index.html: HTML template for the home page and prediction result display.
new_g40.csv: Dataset used for training the machine learning model.
Voting_Classifier_(SVM + Random Forest)_model_data_c1.pkl: Pre-trained machine learning model saved using joblib.

## Dependencies

1. Flask
2. pandas
3. joblib

## Model Details

The application uses a combination of Support Vector Machine (SVM) and Random Forest classifiers for predicting heart disease. The model is loaded from the 'Voting_Classifier_(SVM + Random Forest)_model_data_c1.pkl' file.

## Input Features

1. Age Category
2. Sex
3. BMI (Body Mass Index)
4. Smoking
5. Alcohol Drinking
6. Stroke
7. Physical Health
8. Mental Health
9. Difficulty Walking
10. Diabetic
11. Physical Activity
12. General Health
13. Sleep Time
14. Asthma
15. Kidney Disease
16. Skin Cancer

