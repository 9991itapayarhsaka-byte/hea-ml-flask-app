from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)

# Load trained model
model = joblib.load("hea_model.pkl")

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    Fe = float(request.form["Fe"])
    Ni = float(request.form["Ni"])
    Cr = float(request.form["Cr"])
    Co = float(request.form["Co"])

    features = np.array([[Fe, Ni, Cr, Co]])
    prediction = model.predict(features)[0]

    return render_template("index.html", prediction_text=f"Predicted Hardness: {prediction:.2f}")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
