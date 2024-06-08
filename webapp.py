from flask import Flask, request, render_template, redirect, url_for, send_from_directory, jsonify
from flask_cors import CORS
import os
from werkzeug.utils import secure_filename
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd
import torch
import pytesseract

app = Flask(__name__)
CORS(app) #PNR

# Load the YOLOv5 model - PNR
model_path_PNR = r'runs/train/exp2/weights/best.pt'  # Ensure this path is correct PNR
if not os.path.exists(model_path_PNR):
    model_path_PNR = r'yolov5/runs/train/exp2/weights/best.pt'  # Update this path to the correct one

model_PNR = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path_PNR, force_reload=True) #PNR

# Define a folder to store uploaded images for PNR
UPLOAD_FOLDER_PNR = 'uploads/'
os.makedirs(UPLOAD_FOLDER_PNR, exist_ok=True)
app.config['UPLOAD_FOLDER_PNR'] = UPLOAD_FOLDER_PNR

# Path to Tesseract executable
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'  # Ensure this path is correct

# Set the upload folder and allowed extensions TSR
UPLOAD_FOLDER = 'static/uploads/'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load the model and labels TSR
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

################## TSR START APP ROUTE
@app.route('/detect', methods=['POST'])
def detect():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400

    if file:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER_PNR'], filename)
        file.save(filepath)

        # Perform detection
        results = model_PNR(filepath)

        # Load the image with OpenCV
        img = cv2.imread(filepath)

        # Initialize a list to hold detected license plates
        plate_numbers = []

        # Extract license plate regions and apply OCR
        for det in results.pandas().xyxy[0].itertuples():
            xmin, ymin, xmax, ymax = int(det.xmin), int(det.ymin), int(det.xmax), int(det.ymax)
            plate_region = img[ymin:ymax, xmin:xmax]
            plate_number = pytesseract.image_to_string(plate_region, config='--psm 7')
            plate_numbers.append(plate_number.strip())

            # Draw the bounding box and plate number on the image
            cv2.rectangle(img, (xmin, ymin), (xmax, ymax), (0, 255, 0), 2)
            cv2.putText(img, plate_number.strip(), (xmin, ymin - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Save the annotated image
        annotated_image_path = os.path.join(app.config['UPLOAD_FOLDER_PNR'], 'annotated_' + filename)
        cv2.imwrite(annotated_image_path, img)

        # Verify the file has been saved
        if not os.path.exists(annotated_image_path):
            return jsonify({'error': 'File not saved'}), 500

        # Prepare response
        response = {
            'filename': 'annotated_' + filename,
            'detections': results.pandas().xyxy[0].to_dict(orient='records'),
            'plate_numbers': plate_numbers
        }

        return jsonify(response)

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER_PNR'], filename)
################## TSR END APP ROUTE

if __name__ == '__main__':
    app.run(debug=True)
