import sklearn
import joblib
import pandas as pd
from flask import Flask, request, jsonify, make_response
from pathlib import Path

# les fonctions définies dans la pipeline
def extraire_la_première_lettre(serie):
  return pd.DataFrame(serie.str[0])

# Load Model
model_path = Path('.') / 'model' / 'titanic.model'
model = joblib.load(model_path)

# Start Flask
app = Flask('__name__')

@app.route('/ping', methods=['GET'])
def ping():
  return ('pong', 200)

@app.route('/predict', methods=['POST'])
def predict():
  # print(request)
  # print(request.get_data())
  # print(request.json)
  # Encapsuler dans une dataframe
  df = pd.DataFrame(request.json)
  # Return class
  result = model.predict(df)[0]
  return (str(result), 201)



app.run(host='0.0.0.0', port=8000)