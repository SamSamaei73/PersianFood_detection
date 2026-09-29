import tensorflow as tf
from keras.models import Sequential
from keras.layers import Flatten, Dense, Dropout, Activation
import matplotlib.pyplot as plt
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Function to load your dataset
def load_data(data_path):
    x_train, y_train, x_test, y_test = [], [], [], []
    # Implement the logic to load and split your data into x_train, y_train, x_test, y_test
    # This is a placeholder; you need to implement this based on your dataset structure.
    return (np.array(x_train), np.array(y_train)), (np.array(x_test), np.array(y_test))

P_datasets = './Datasets/Persian_Alphabet/data/data'

# Load the data
(x_train, y_train), (x_test, y_test) = load_data(P_datasets)

# Debugging: print shapes of the data
print("x_train shape:", x_train.shape)
print("y_train shape:", y_train.shape)
print("x_test shape:", x_test.shape)
print("y_test shape:", y_test.shape)

# Ensure the data has the correct dimensions
if x_train.ndim == 1:
    x_train = np.expand_dims(x_train, axis=-1)
if x_test.ndim == 1:
    x_test = np.expand_dims(x_test, axis=-1)

# Normalize the data
x_train = tf.keras.utils.normalize(x_train, axis=-1)
x_test = tf.keras.utils.normalize(x_test, axis=-1)

# Build the model
model = Sequential()
model.add(Flatten(input_shape=x_train.shape[1:]))  # specify input shape
model.add(Dense(128, activation='relu'))
model.add(Dense(256, activation='relu'))
model.add(Dense(512, activation='relu'))
model.add(Dense(len(set(y_train)), activation='softmax'))  # Use 'softmax' for multi-class classification

# Compile the model
model.compile(loss='sparse_categorical_crossentropy', metrics=['accuracy'], optimizer='adam')

# Train the model
cnn_history = model.fit(x_train, y_train, epochs=10, batch_size=32, validation_data=(x_test, y_test))

# Save the model
model.save('Persian_Alphabet.h5')

# Plot training & validation accuracy values
plt.plot(cnn_history.history['accuracy'])
plt.plot(cnn_history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()

# Plot training & validation loss values
plt.plot(cnn_history.history['loss'])
plt.plot(cnn_history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='best')
plt.show()
