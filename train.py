import matplotlib.pyplot as plt
from data_preparation import load_labels, load_data, preprocess_data, create_data_generator
from model import create_model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# Define the path to your dataset
data_dir = './traffic_data/data'
labels_file = 'labels.csv'

# Load and preprocess data
class_names = load_labels(labels_file)
print("Loaded labels:", class_names)  # Debug statement
images, labels = load_data(data_dir, class_names)
X_train, X_test, y_train, y_test = preprocess_data(images, labels)

# Create the data generator
train_generator = create_data_generator(X_train, y_train)

# Create the model
input_shape = (64, 64, 3)
num_classes = len(class_names)
model = create_model(input_shape, num_classes)

# Define callbacks
early_stopping = EarlyStopping(monitor='val_loss', patience=10, verbose=1, restore_best_weights=True)
model_checkpoint = ModelCheckpoint('traffic_sign_model.keras', monitor='val_loss', save_best_only=True, verbose=1)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, verbose=1)

# Increase the number of epochs
max_epochs = 30

# Train the model
history = model.fit(
    train_generator,
    epochs=max_epochs,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, model_checkpoint, reduce_lr]
)

# Plot training history
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val_accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.ylim([0, 1])
plt.legend(loc='lower right')
plt.show()

# Evaluate the model
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f'Test accuracy: {test_acc}')
