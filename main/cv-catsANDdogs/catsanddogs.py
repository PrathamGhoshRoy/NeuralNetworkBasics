import tensorflow as tf 
import numpy as np 
import matplotlib.pyplot as plt 
import pickle
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten 
from tensorflow.keras.layers import Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import TensorBoard
import time

import warnings
warnings.filterwarnings('ignore')

model_name = f'Cats-Vs-Dogs-CNN-64x2-{int(time.time())}'

tensorboard = TensorBoard(log_dir=f'logs/{model_name}')

pickle_in = open("Jupyter Notebooks\X.pickle", "rb")
X = pickle.load(pickle_in)

pickle_in = open("Jupyter Notebooks\y.pickle", "rb")
y = pickle.load(pickle_in)

X = X/255.0

model = Sequential()

model.add(Conv2D(64, (3,3), input_shape = X.shape[1:]))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten()) # Flattening the 3D features to a 1D feature vector

# Output Layer
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(optimizer='adam',
             loss='binary_crossentropy',
             metrics=['accuracy'])
                      
model.fit(X, y, batch_size = 32, validation_split=0.3, epochs=6, callbacks=[tensorboard])

