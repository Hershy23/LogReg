import pickle
from flask import Flask, request, render_template, jsonify


with open('logistic_model_l2.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        input_features = [float(x) for x in request.form['features'].split(',')]
        if len(input_features) != 54:
            return jsonify({'error': 'Please enter exactly 54 comma-separated values.'})

        prediction = model.predict([input_features])[0]
        return jsonify({'prediction': 'Spam' if prediction == 1 else 'Not Spam'})
    except Exception as e:
        return jsonify({'error': f'Invalid input: {str(e)}'})

if __name__ == '__main__':
    app.run(debug=True)
