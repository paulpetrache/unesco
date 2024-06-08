import os
import numpy as np
import cv2
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def load_labels(labels_file):
    labels_df = pd.read_csv(labels_file, header=0, index_col=0)  # Skip the header row
    return labels_df['Name'].to_dict()  # Access the 'Name' column

def load_data(data_dir, labels):
    images = []
    image_labels = []
    for label in labels:
        class_dir = os.path.join(data_dir, str(label))
        if not os.path.isdir(class_dir):
            print(f"Warning: Directory {class_dir} does not exist. Skipping...")
            continue
        for image_name in os.listdir(class_dir):
            image_path = os.path.join(class_dir, image_name)
            if not os.path.isfile(image_path):
                continue
            image = cv2.imread(image_path)
            image = cv2.resize(image, (64, 64))  # Resize to a fixed size
            images.append(image)
            image_labels.append(label)
    images = np.array(images)
    image_labels = np.array(image_labels)
    return images, image_labels

def preprocess_data(images, labels):
    images = images / 255.0
    labels = to_categorical(labels)
    X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

def create_data_generator(X_train, y_train):
    datagen = ImageDataGenerator(
        rotation_range=10,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
        horizontal_flip=False,
        fill_mode='nearest'
    )
    datagen.fit(X_train)
    return datagen.flow(X_train, y_train, batch_size=32)
