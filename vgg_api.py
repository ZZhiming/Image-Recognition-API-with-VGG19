# Importing flask module in the project is mandatory
# An object of Flask class is our WSGI application.
import os
from flask import Flask, flash, request, redirect, url_for, render_template, jsonify
from werkzeug.utils import secure_filename
import json
import time
import cv2
import numpy as np
import vgg_prediction


app = Flask(__name__)


@app.route('/', methods=['GET'])
def index():
    return "Welcome to Object Detection!"

# Calls VGG model for classification
@app.route('/prediction', methods=['POST'])
def prediction():
    data = request.json
    arr = np.array(data['img_arr'])
    predictions = vgg_prediction.predict(arr)
    return json.dumps(predictions)



if __name__ == '__main__':
    app.secret_key = 'aksldfkj'
    # run() method of Flask class runs the application
    # on the local development server.
    app.run(host='0.0.0.0', port=50001, debug=True)


