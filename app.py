from flask import Flask, request, jsonify, render_template
import pickle
import numpy as np

app = Flask(__name__)

# Load the saved model and scaler
model = pickle.load(open("FinalRFModel.pkl", 'rb'))
scaler = pickle.load(open("scaler.pkl", 'rb'))

@app.route('/')
def home():
    return render_template('index.html')  # Save the updated HTML as 'templates/index.html'

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        form_data = request.form

        

        # Prepare features array
        features = np.array([[
            float(form_data['amt']),
            int(form_data['zip']),  
            float(form_data['City_pop']),
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
