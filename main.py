from flask import Flask, render_template, request
import pandas as pd
import joblib
#from transformers.transformers import AgeCategoryTransformer



app = Flask(__name__)

# Load the Naive Bayes model
model = joblib.load('Voting_Classifier_(SVM + Random Forest)_model_data1_new.pkl')

# Load the dataset
df = pd.read_csv('new_g40.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get the form input
    AgeCategory = int(request.form['AgeCategory'])
    Sex = int(request.form['Sex'])
    BMI = float(request.form['bmi'])
    Smoking = int(request.form['smoking'])
    AlcoholDrinking = int(request.form['alcohol'])
    Stroke = int(request.form['stroke'])
    PhysicalHealth = int(request.form['physical_health'])
    MentalHealth = int(request.form['mental_health'])
    DiffWalking = int(request.form['DiffWalking'])
    Diabetic = int(request.form['diabetic'])
    PhysicalActivity = int(request.form['physically_active'])
    GenHealth = int(request.form['general_health'])
    SleepTime = float(request.form['sleep_time'])
    Asthma = int(request.form['Asthma'])
    KidneyDisease = int(request.form['KidneyDisease'])
    SkinCancer = int(request.form['SkinCancer'])

    # Create a DataFrame with the input data
    input_data = pd.DataFrame({
        'AgeCategory': [AgeCategory],
        'Sex': [Sex],
        'BMI': [BMI],
        'Smoking': [Smoking],
        'AlcoholDrinking': [AlcoholDrinking],
        'Stroke': [Stroke],
        'PhysicalHealth': [PhysicalHealth],
        'MentalHealth': [MentalHealth],
        'DiffWalking': [DiffWalking],
        'Diabetic': [Diabetic],
        'PhysicalActivity': [PhysicalActivity],
        'GenHealth': [GenHealth],
        'SleepTime': [SleepTime],
        'Asthma': [Asthma],
        'KidneyDisease': [KidneyDisease],
        'SkinCancer': [SkinCancer],
    })

    # Make a prediction
    prediction = model.predict(input_data)

    # Determine the prediction message
    if prediction[0] == 1:
        result = 'Positive for Heart Disease'
    else:
        result = 'Negative for Heart Disease'

    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
