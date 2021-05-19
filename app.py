#Import neccessary modules
import pickle
import string
import numpy as np
from flask import Flask, request, jsonify, render_template

app = Flask(__name__)

#Load the model 
model = pickle.load(open('model.pkl', 'rb'))

#When initially taken to the local server website, the template will be rendered
@app.route('/')
def home():
    return render_template('index.html')

#When the 'Submit' button is pressed, this function will activate
@app.route('/predict',methods=['POST'])
def predict():
    #Clean headline from the text
    text = str(request.form['headline'])
    text = text.strip().lower()
    text = text.translate(str.maketrans('', '', string.punctuation))
    #Get the probabilities for both the headline being real(0) and the headline being from the Onion(1) 
    pred_probs = np.ravel(model.predict_proba([text]))
    if pred_probs[0] > pred_probs[1]:
        prediction = 'Real'
        chance = int(pred_probs[0] * 100)
    else:
        prediction = 'The Onion'
        chance = int(pred_probs[1] * 100)

    #Return the html template with a string variable prediction which gives the information off of the model prediction 
    return render_template('index.html', prediction='{}% chance of the headline being {}'.format(chance, prediction))

#Run the app
if __name__ == "__main__":
    app.run(debug=True)
