import tensorflow as tf
from keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt

# Load the pre-trained model
new_model = load_model('number_model.h5')

# Load the MNIST dataset
mnist = tf.keras.datasets.mnist
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Normalize the test dataset
x_test = tf.keras.utils.normalize(x_test, axis=1)

# Select a random index
random_index = np.random.randint(0, len(x_test))

# Get the random image and its corresponding true label
random_image = x_test[random_index]
random_label = y_test[random_index]

# Reshape the image to match the model's expected input
random_image_reshaped = random_image.reshape(1, 28, 28, 1)  # If your model expects a different shape, adjust accordingly

# Make a prediction using the model
prediction = new_model.predict(random_image_reshaped)

# Print the predicted and true label
predicted_label = np.argmax(prediction)
print(f"Predicted label: {predicted_label}")
print(f"True label: {random_label}")

# Display the random image
plt.imshow(random_image, cmap='gray')
plt.title(f"Predicted: {predicted_label}, True: {random_label}")
plt.show()
