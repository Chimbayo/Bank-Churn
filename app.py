from flask import Flask, render_template, request, jsonify
import joblib
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load your trained model
model = joblib.load('lgb_model.pkl')  # Adjust path if needed

@app.route('/')
def home():
    return render_template('login page.html')  # Correct HTML filename
@app.route('/about')
def about():
    return render_template('about page.html')  # About page route

@app.route('/contacts')
def contacts():
    return render_template('contacts.html')  # Contacts page route
@app.route('/login')
def login():
    return render_template('login page.html')  # login page route
@app.route('/dashboard')
def dashboard():
    return render_template('dashboard page.html')  # login page route
@app.route('/uploadcsv')
def uploadcsv():
    try:
        file = request.files['file']
        df = pd.read_csv(file)

        # Ensure columns match model's expected order
        features = df[[
            'Age', 'Gender', 'District', 'Region', 'Location_Type', 'Customer_Type', 'Employment_Status', 
            'Income_Level', 'Education_Level','Tenure', 'Balance', 'Credit_Score', 'Outstanding_Loans', 
            'Num_Of_Products', 'Mobile_Banking_Usage', 'Number_of_Transactions_per/Month', 'Num_Of_Complaints',
            'Proximity_to_NearestBranch_or_ATM (km)', 'Mobile_Network_Quality', 'Owns_Mobile_Phone'
        ]].values

        predictions = model.predict(features) # type: ignore

        # Optional: get probability if model supports it
        try:
            probabilities = model.predict_proba(features) # type: ignore
            results = [
                {'prediction': int(pred), 'probability': round(float(np.max(prob)) * 100, 2)}
                for pred, prob in zip(predictions, probabilities)
            ]
        except AttributeError:
            results = [{'prediction': int(pred)} for pred in predictions]

        return jsonify({'predictions': results})

    except Exception as e:
        return jsonify({'error': str(e)})
    return render_template('uploadcsv.html')  # login page route


@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()

        # Collect and prepare features
        features = np.array([
            int(data['Age']),   
            int(data['Gender']),
            int(data['District']),
            int(data['Region']),
            int(data['Location_Type']),
            int(data['Customer_Type']),
            int(data['Employment_Status']),
            int(data['Income_Level']),
            int(data['Education_Level']),
            int(data['Tenure']),
            int(data['Balance']),
            int(data['Credit_Score']),
            int(data['Outstanding_Loans']),
            int(data['Num_Of_Products']),
            int(data['Mobile_Banking_Usage']),
            int(data['Number_of _Transactions_per/Month']),
            int(data['Num_Of_Complaints']),
            float(data['Proximity_to_NearestBranch_or_ATM (km)']),
            int(data['Mobile_Network_Quality']),
            int(data['Owns_Mobile_Phone'])
        ]).reshape(1, -1)

        # Predict using XGBoost model
        probabilities = model .predict(features)[0] # type: ignore

        # If model outputs single probability
        if len(probabilities.shape) == 1:
            probability = round(float(probabilities[0]) * 100, 2)
            prediction = 1 if probabilities[0] >= 0.5 else 0
        else:
            probability = round(float(np.max(probabilities)) * 100, 2)
            prediction = np.argmax(probabilities)

        result = "Customer will stay" if prediction == 0 else "Customer will leave"

        return jsonify({'prediction': result, 'probability': probability})

    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)})


if __name__ == '__main__':
    app.run(debug=True)