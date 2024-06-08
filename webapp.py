from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import torch
#import pytesseract

app = Flask(__name__)

# Set the upload folder and allowed extensions
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model and labels
model = tf.keras.models.load_model('traffic_sign_model.keras')
labels_file = 'labels.csv'
class_names = pd.read_csv(labels_file, header=0, index_col=0)['Name'].to_dict()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def predict_traffic_sign(image_path, model, class_names):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64)) / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    return class_names.get(predicted_class, "Unknown class")

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        # Check if the post request has the file part
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        # If user does not select file, browser also
        # submit an empty part without filename
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            prediction = predict_traffic_sign(file_path, model, class_names)
            return render_template('index.html', prediction=prediction, filename=filename)
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
