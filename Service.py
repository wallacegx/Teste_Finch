from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
import os
import numpy as np

saida = 'Saida/'

app = Flask(__name__)
CORS(app)


@app.route('/')
def root():
    return "Server no ar"


@app.route('/letra', methods=['POST'])
def classify():
    file = request.json

    letra = file["letra"]
    model = joblib.load(saida + 'classificator.pkl')

    genero = model.predict([letra])

    data = {"Genero": genero[0]}
    response = jsonify(data)
    return response


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)