from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler
model = pickle.load(open(r"C:\Users\tiwar\Desktop\ml_deploy\FinalRFModel.pkl", 'rb'))
scaler = pickle.load(open(r"C:\Users\tiwar\Desktop\ml_deploy\scaler.pkl", 'rb'))
encoders = pickle.load(open(r"C:\Users\tiwar\Desktop\ml_deploy\encoder.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  # Save the updated HTML as 'templates/index.html'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form

        # Encode categorical data using the LabelEncoders
        encoded_category = encoders['category'].transform([form_data['category']])[0]
        encoded_gender = encoders['gender'].transform([form_data['gender']])[0]
        encoded_state = encoders['state'].transform([form_data['state']])[0]
        encoded_day_name = encoders['day_name'].transform([form_data['day_name']])[0]

        # Prepare features array
        features = np.array([[
            encoded_category,
            float(form_data['amt']),
            encoded_gender,
            encoded_state,
            int(form_data['zip']),  # Assuming ZIP is numeric
            float(form_data['City_pop']),
            encoded_day_name,
            float(form_data['Hour']),
            float(form_data['distance_from_home']),
            float(form_data['age'])
        ]])

        # Scale the features
        features_scaled = scaler.transform(features)

        # Predict using the model
        prediction = model.predict(features_scaled)
        value="Fraud" if prediction[0]==1 else "Non-Fraud"

        return render_template("index.html", prediction_text="The Transaction is {}".format(value))
    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    app.run(debug=True)