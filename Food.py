import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

train_dir = './Food/Train'
validation_dir = './Food/Validation'

train_gen = ImageDataGenerator(rescale=1./255 , rotation_range=30, width_shift_range=0.2, height_shift_range=0.2 , shear_range=0.2, zoom_range=0.2, horizontal_flip=True , fill_mode='nearest')
validation_gen = ImageDataGenerator(rescale=1./255)

train_gen = train_gen.flow_from_directory(directory=train_dir,target_size=(300,300),batch_size=10,class_mode='categorical')
validation_gen = validation_gen.flow_from_directory(directory=validation_dir,target_size=(300,300),batch_size=10,class_mode='categorical')

def sample_image(image_array):
    fig, axes = plt.subplots(1, 5, figsize=(20, 20))
    axes = axes.flatten()
    for img, ax in zip(image_array, axes):
        ax.imshow(img)
    plt.tight_layout()
    plt.show()

images, labels = next(train_gen)
sample_image(images[:5])

model = Sequential()


model.add(Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same', input_shape=(300,300,3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(256, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dense(3, activation='softmax'))

model.summary()

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(train_gen,steps_per_epoch=100 ,epochs=10, validation_data=validation_gen , validation_steps=50)
model.save('Food.h5')

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

# Visualizing model loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()
