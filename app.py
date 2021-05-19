import pickle
import string
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

model = pickle.load(open('C:/Users/arnie2014/Desktop/model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
    text = str(request.form['headline'])
    text = text.strip().lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    pred_probs = np.ravel(model.predict_proba([text]))
    if pred_probs[0] > pred_probs[1]:
        prediction = 'Real'
        chance = int(pred_probs[0] * 100)
    else:
        prediction = 'The Onion'
        chance = int(pred_probs[1] * 100)

    return render_template('index.html', prediction='{}% chance of the headline being {}'.format(chance, prediction))

if __name__ == "__main__":
    app.run(debug=True)
