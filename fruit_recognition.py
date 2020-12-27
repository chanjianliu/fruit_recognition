import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
import tensorflow as tf
from PIL import Image
import glob

def load_img(x):
    image = Image.open(x).convert('RGB')
    resized = image.resize((100,100))
    image = np.array(resized)
    image = np.reshape(image, (1, 100, 100, 3))
    return image

# may need to change path to access data files
train_img = glob.glob('train/*.jpg')
test_img = glob.glob('test/*.jpg')

x_train = []
y_train = []
x_test = []
y_test = []
count = 1

for fruit in ['apple', 'orange', 'banana', 'mixed']:
    number = 1
    for i in train_img:
        if fruit in i:
            print(fruit, number)
            number += 1
            if x_train == []:
                x_train = [load_img(i)]
                y_train += [0 if fruit == 'apple' else (1 if fruit == 'orange' else (2 if fruit == 'banana' else 3))]
            else:
                x_train = np.append(x_train, [load_img(i)], axis=0)
                y_train += [0 if fruit == 'apple' else (1 if fruit == 'orange' else (2 if fruit == 'banana' else 3))]
    for i in test_img:
        if fruit in i:
            if x_test == []:
                x_test = [load_img(i)]
                y_test += [0 if fruit == 'apple' else (1 if fruit == 'orange' else (2 if fruit == 'banana' else 3))]
            else:
                x_test = np.append(x_test, [load_img(i)], axis=0)
                y_test += [0 if fruit == 'apple' else (1 if fruit == 'orange' else (2 if fruit == 'banana' else 3))]

x_train = np.reshape(x_train, (x_train.shape[0], 100, 100, 3))
x_test = np.reshape(x_test, (x_test.shape[0], 100, 100, 3))

# scaling data to a range between 0 and 1
x_train = x_train / 255
x_test = x_test / 255

# encoding the 4 category (apple, orange, banana, mixed)
y_train = tf.keras.utils.to_categorical(y_train, 4)
y_test = tf.keras.utils.to_categorical(y_test, 4)

print(x_train.shape)
print(y_train.shape)
print(x_test.shape)
print(y_test.shape)

# this is the training method declaration
def run_cnn(_x_train, _y_train, _x_test, _y_test):
	model = tf.keras.Sequential()
	model.add(tf.keras.layers.Conv2D(32, (3, 3), 
		activation='relu', input_shape=(100, 100, 3)))
	model.add(tf.keras.layers.Conv2D(32, (3, 3), 
		activation='relu'))
	model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2)))
	model.add(tf.keras.layers.Dropout(0.25))
	model.add(tf.keras.layers.Flatten())	
	model.add(tf.keras.layers.Dense(128, activation='relu'))
	model.add(tf.keras.layers.Dropout(0.5))
	model.add(tf.keras.layers.Dense(4, activation='softmax'))
	model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
	model.summary()

	history = model.fit(_x_train, _y_train, 
		batch_size=24, epochs=10, verbose=1,
		validation_data=(_x_test, _y_test))
		
	score = model.evaluate(_x_test, _y_test)
	print("score =", score)

# this is the command to run the training
run_cnn(x_train, y_train, x_test, y_test)