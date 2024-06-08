import os
import pandas as pd

def load_labels(labels_file):
    labels_df = pd.read_csv(labels_file, header=0, index_col=0)  # Skip the header row
    print(labels_df.head())  # Debug statement to check DataFrame contents
    print(labels_df.columns)  # Debug statement to check DataFrame columns
    return labels_df.iloc[:, 0].to_dict()  # Return a dictionary mapping class index to class name

if __name__ == '__main__':
    labels_file = 'labels.csv'
    class_names = load_labels(labels_file)
    print("Class names loaded:", class_names)  # Debug statement
