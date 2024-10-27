from flask import Flask, render_template, request
import numpy as np
import joblib
import pickle


# Initialize Flask app
app = Flask(__name__)


with open('co2_emission_model.pkl', 'rb') as f:
    catboost_model = pickle.load(f)

@app.route("/")
def home():
    return render_template("index.html")  
@app.route("/form")
def form():
    return render_template("form.html") 

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Extract form data
     make = request.form['make']
     model_name = request.form['model']
     vehicle_type = request.form['vehicle_type']
     engine_size = float(request.form['engine_size'])
     cylinders = int(request.form['cylinders'])
     transmission_type = request.form['transmission_type']
     fuel_type = request.form['fuel_type']
     fuel_consumption_city = float(request.form['fuel_consumption_city'])
     fuel_consumption_hwy = float(request.form['fuel_consumption_hwy'])
     fuel_consumption_comb = float(request.form['fuel_consumption'])
     fuel_consumption_comb_mpg = float(request.form['fuel_consumptionmpg'])
    
    # Prepare the data in the format that the model expects
    input_features = np.array([[make, model_name, vehicle_type, engine_size, cylinders, transmission_type, 
                            fuel_type, fuel_consumption_city, fuel_consumption_hwy, fuel_consumption_comb,fuel_consumption_comb_mpg]])


    print("Input Features:", input_features)
    # Predict using the loaded model
    prediction = catboost_model.predict(input_features)

        # Return the prediction result to the HTML page
    return render_template('result.html', emission=round(prediction[0], 2))

# Run the Flask app
if __name__ == "__main__":
    app.run(debug=True)
