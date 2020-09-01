from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def predict():
    df = pd.read_csv(request.files['file'])
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
        x = df.drop('Unnamed: 0', axis=1)
        y = model.predict(x)
    return render_template('results.html', data=y)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8080, debug=True)