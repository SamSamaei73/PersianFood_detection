import numpy as np
import matplotlib.pyplot as plt
import os
import cv2

AnimalData = './Animal/train'
Categories = ['cat', 'dog' , 'Elephant' , 'Horse' , 'Lion']

for category in Categories:
    path = os.path.join(AnimalData, category)
    for img in os.listdir(path):
        img_array = cv2.imread(os.path.join(path, img) , cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array , cmap='gray')
        plt.show()
        break
    break

