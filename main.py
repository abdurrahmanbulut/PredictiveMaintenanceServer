from flask import Flask, request, jsonify
import pickle
import numpy as np
from flask_cors import cross_origin, CORS
import joblib
from sklearn.preprocessing import LabelEncoder
import pandas as pd
from sklearn.preprocessing import PolynomialFeatures
from sklearn.preprocessing import StandardScaler
import time


app = Flask(__name__)
CORS(app)
# Load the saved model
loaded_model = joblib.load('./knn_model.pkl')
failure = ['comp0', 'comp1', 'comp2', 'comp3', 'comp4']
label_encoder = LabelEncoder()
# Fit the label encoder on the "failure" column
label_encoder.fit(failure)
poly = PolynomialFeatures(degree=2)
# Standardize the features
scaler = StandardScaler() 


def foo(arr):
    c0, c1, c2, c3, c4 = 0, 0, 0, 0, 0
    # Apply Polynomial Features
    X = poly.fit_transform(arr)
    X = scaler.fit_transform(X)
    start_time = time.time()  # Başlangıç zamanını kaydet
    predicted_labels = loaded_model.predict(X)
    end_time = time.time()  # Bitiş zamanını kaydet
    total_time = (end_time - start_time) * 1000  # Süreyi milisaniyeye dönüştür
    print("Fonksiyon süresi: {} ms".format(total_time))

    predicted_classes = label_encoder.inverse_transform(predicted_labels)
    for i in predicted_classes:
        if i == "comp0":
            c0 += 1
        if i == "comp1":
            c1 += 1
        if i == "comp2":
            c2 += 1
        if i == "comp3":
            c3 += 1
        if i == "comp4":
            c4 += 1    
    return c0, c1, c2, c3, c4


@app.route('/predict', methods=['POST'])
def predict():
    data = request.json  # get data from POST request
    df = pd.DataFrame(data)
    c0, c1, c2, c3, c4 = foo(df)
    counts = {"c0": c0, "c1": c1, "c2": c2, "c3": c3, "c4": c4}  # Create a JSON object with the counts
    return jsonify({'prediction': "Çalıştı", 'counts': counts})  # Return the JSON object as the response

if __name__ == '__main__':
    app.run(port=5000)
