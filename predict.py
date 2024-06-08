import os
import cv2
import numpy as np
import tensorflow as tf
import pandas as pd

def load_labels(labels_file):
    labels_df = pd.read_csv(labels_file, header=0, index_col=0)  # Skip the header row
    return labels_df['Name'].to_dict()  # Access the 'Name' column

def predict_traffic_sign(image_path, model, class_names):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (64, 64)) / 255.0
    image = np.expand_dims(image, axis=0)
    predictions = model.predict(image)
    predicted_class = np.argmax(predictions)
    print(f"Predicted class index: {predicted_class}")  # Debug statement
    if predicted_class not in class_names:
        print(f"Warning: Predicted class index {predicted_class} not found in class_names.")
    return class_names.get(predicted_class, "Unknown class")

if __name__ == '__main__':
    labels_file = 'labels.csv'
    model = tf.keras.models.load_model('traffic_sign_model.keras')
    class_names = load_labels(labels_file)
    print("Class names loaded:", class_names)  # Debug statement
    test_dir = './traffic_data/test'  # Set to your test directory
    for image_name in os.listdir(test_dir):
        image_path = os.path.join(test_dir, image_name)
        predicted_class = predict_traffic_sign(image_path, model, class_names)
        print(f'The predicted class for {image_name} is: {predicted_class}')
