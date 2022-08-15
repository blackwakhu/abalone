from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

model = pickle.load(open('linear_regression.pkl', 'rb'))

@app.route('/')
def homepage():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    initial_features = [int(x) for x in request.form.values()]
    final_features = np.array(initial_features).reshape(1,-1)
    prediction = model.predict(final_features)
    return render_template('index.html', prediction=f'the ring size is {prediction}')

if __name__ == '__main__':
    app()