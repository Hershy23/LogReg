import pickle
import numpy as np
from flask import Flask, request, jsonify, render_template

with open("logistic_model_none.pkl", "rb") as file:
    model_none = pickle.load(file)

with open("logistic_model_l2.pkl", "rb") as file:
    model_l2 = pickle.load(file)

app = Flask(__name__)

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["features"] 
        input_data = np.array(data).reshape(1, -1)

        pred_none = model_none.predict(input_data)[0]
        pred_l2 = model_l2.predict(input_data)[0]
        prob_none = model_none.predict_proba(input_data)[0][1]
        prob_l2 = model_l2.predict_proba(input_data)[0][1]

        response = {
            "No Regularization Prediction": "Spam" if pred_none == 1 else "Not Spam",
            "No Regularization Probability": f"{prob_none:.4f}",
            "L2 Regularization Prediction": "Spam" if pred_l2 == 1 else "Not Spam",
            "L2 Regularization Probability": f"{prob_l2:.4f}"
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == "__main__":
    app.run(debug=True)
