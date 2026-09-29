import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import os
import random


# 2. The second script should load the trained model and predict the hand sign in the image.
def load_and_prepare_image(image_path, target_size=(300, 300)):
    img = tf.keras.preprocessing.image.load_img(image_path, target_size=target_size)
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array_expanded_dims = np.expand_dims(img_array, axis=0)
    return img_array_expanded_dims / 255.


# a) The tested image is to be supplied via the arguments list
def predict_image(model_path, image_path):
    model = tf.keras.models.load_model(model_path)
    img_prepared = load_and_prepare_image(image_path)

    predictions = model.predict(img_prepared)
    predicted_class_index = np.argmax(predictions, axis=1)
    predicted_score = np.max(predictions)

    # If you have another method for mapping class indices to class names, change this section.
    class_labels = ['Gheymeh', 'Ghormesabzi', 'Kabab']
    predicted_label = class_labels[predicted_class_index[0]]

    img = plt.imread(image_path)
    plt.imshow(img)
    plt.title(f"Predicted: {predicted_label}, Score: {predicted_score:.2f}")
    plt.show()
    return predicted_label, predicted_score

# b) visualisation of the supplied image with the prediction score and predicted label

if __name__ == "__main__":

    model_path= './Food.h5'
    directory = './Food/Test'
    png_files = [file for file in os.listdir(directory) if file.endswith('.jpeg')]
    random_png = random.choice(png_files)
    image_path = os.path.join(directory, random_png)
    predict_image(model_path, image_path)