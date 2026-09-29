import tensorflow as tf
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense , Dropout , Flatten , Dense

mnist = tf.keras.datasets.mnist

# we divide our datasets to two test and train group
(x_train , y_train) , (x_test , y_test ) = mnist.load_data()

#Normalitation
x_train =tf.keras.utils.normalize(x_train)
x_test =  tf.keras.utils.normalize(x_test)

#make our model
model = Sequential()
model.add(Flatten(input_shape=(28,28)))
model.add(Dense(128 , activation='relu'))
model.add(Dense(128 , activation="relu"))
model.add(Dense(10, activation='softmax'))

model.compile(optimizer='adam' ,loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(x_train , y_train , epochs=3)

val_loss, val_acc = model.evaluate(x_test , y_test )
print(val_loss, val_acc)
# cmap means color map whe use it for change colourful image to gray or binary
# plt.imshow(x_train[2] , cmap=plt.cm.binary)
# plt.show()
# print(x_train[2])

model.save('number_model.h5')

