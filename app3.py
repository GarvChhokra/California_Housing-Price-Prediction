# -*- coding: utf-8 -*-
"""
Created on Sat Feb 18 19:06:09 2023

@author: kmerl
"""

import numpy as np
from flask import Flask, request, render_template
import joblib
import pickle

import pandas as pd

app = Flask(__name__)

model = joblib.load("my_model.pkl")

pipeline = joblib.load("my_pipeline.pkl")
cols = ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population','households','median_income','ocean_proximity']

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/result", methods=["POST"])
def result():
       
    longitude = np.array([request.form['longi']])
    latitude = np.array([request.form['lati']])
    housing_median_age = np.array([request.form['housing_median_age']])
    total_rooms = np.array([request.form['total_rooms']])
    total_bedrooms = np.array([request.form['total_bedrooms']])
    population = np.array([request.form['population']])
    households = np.array([request.form['households']])
    median_income = np.array([request.form['median_income']])
    ocean_proximity = np.array([request.form['ocean_proximity']])
    final = np.concatenate([longitude,latitude,housing_median_age,total_rooms,
                           total_bedrooms,population,households,median_income,
                             ocean_proximity])
    final = np.array(final)
    data = pd.DataFrame([final], columns=cols)
    data_trans = pipeline.transform(data)
    prediction = model.predict(data_trans)
    return render_template("result.html", prediction = round(prediction[0],3))

if __name__ == "__main__":
    app.run(debug=True, port=5000)









