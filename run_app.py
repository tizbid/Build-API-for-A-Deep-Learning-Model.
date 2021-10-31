import tensorflow as tf
from tensorflow import keras
from flask import Flask, request, jsonify
import pandas as pd
import numpy as np


# initialize our Flask application and the Keras model
app = Flask(__name__)
model = None
pred_text = input('Enter test text :')

def load_model():
    # load the pre-trained Keras model
    global model
    model  = keras.models.load_model("classifier_model")
    return
    


@app.route("/predict")
def predict():
    result = model.predict([pred_text])
    predict = result[0][0]
    result = [predict, "ham" if predict <0.5 else "spam"]    
    jresult = {pred_text:str(result[1])}
    # return the data dictionary as a JSON response
    return jsonify(jresult)

if __name__ == "__main__":
    print(("* Loading Keras model and Flask starting server..."
        "please wait until server has fully started"))
    load_model()
    app.run()

