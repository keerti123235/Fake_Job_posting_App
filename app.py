import numpy as np
from flask import Flask, request, render_template
import joblib
from helpers import *

model = joblib.load('model.joblib')
vectorizer = joblib.load('vectorizer.joblib')

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():

    features = request.form['job_title'] + ' ' + request.form['location'] + ' ' + request.form['department'] + ' ' + request.form['company_profile'] + ' ' + request.form['description'] + ' ' + request.form['requirements']
    features = features +  request.form['benefits'] + ' ' + request.form['employment_type'] + ' ' + request.form['required_experience']+ ' ' + request.form['required_education'] + ' ' + request.form['industry'] + ' ' + request.form['function']

    inp = get_model_input(features, vectorizer)
    output = model.predict(inp)

    prediction_text = 'fake' if output == 1 else 'real'
    return render_template('results.html', prediction_text=prediction_text)

@app.route('/how',methods=['GET'])
def how():
	return render_template('how.html')


if __name__ == "__main__":
	app.run(debug=True)