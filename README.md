# Convolutional Deep Neural Network for Digit Classification

## AIM

To Develop a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images.

## Problem Statement and Dataset

## Neural Network Model
![image](https://user-images.githubusercontent.com/112486797/192814336-5ec45b38-57c1-400e-9ad1-b41ef7b35571.png)


Include the neural network model diagram.

## DESIGN STEPS

### STEP 1:
Import tensorflow and preprocessing libraries

### STEP 2:
Build a CNN model

### STEP 3:
Compile and fit the model and then predict
```
## PROGRAM

import numpy as np
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.datasets import mnist
import tensorflow as tf
import matplotlib.pyplot as plt
from tensorflow.keras import utils
import pandas as pd
from sklearn.metrics import classification_report,confusion_matrix
from tensorflow.keras.preprocessing import image

(X_train, y_train), (X_test, y_test) = mnist.load_data()

X_train_scaled = X_train/255.0
X_test_scaled = X_test/255.0

y_train_onehot = utils.to_categorical(y_train,10)
y_test_onehot = utils.to_categorical(y_test,10)

X_train_scaled = X_train_scaled.reshape(-1,28,28,1)
X_test_scaled = X_test_scaled.reshape(-1,28,28,1)

np.unique(y_test)

model = keras.Sequential([
    tf.keras.layers.Conv2D(32,kernel_size=3,activation="relu",padding="same"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32,kernel_size=3,activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Conv2D(32,kernel_size=3,activation="relu"),
    tf.keras.layers.MaxPool2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(10,activation="softmax")
])

model.compile(loss="categorical_crossentropy", metrics='accuracy',optimizer="adam")

model.fit(X_train_scaled ,y_train_onehot, epochs=2,
          batch_size=64, 
          validation_data=(X_test_scaled,y_test_onehot))

pd.DataFrame(model.history.history).plot()

x_test_predictions = np.argmax(model.predict(X_test_scaled), axis=1)

confusion_matrix(y_test,x_test_predictions)

print(classification_report(y_test,x_test_predictions))

img = image.load_img('imagefive.jpg')
img_tensor = tf.convert_to_tensor(np.asarray(img))
img_28 = tf.image.resize(img_tensor,(28,28))
img_28_gray = tf.image.rgb_to_grayscale(img_28)
img_28_gray_scaled = img_28_gray.numpy()/255.0

np.argmax(model.predict(img_28_gray_scaled.reshape(1,28,28,1)),axis=1)
```

## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

![image](https://user-images.githubusercontent.com/112486797/192815382-ca3030f3-b494-4943-b14e-8e5613d2b056.png)


### Classification Report

![image](https://user-images.githubusercontent.com/112486797/192815579-59eff797-07a5-4732-9dfd-066313d430f8.png)


### Confusion Matrix
![image](https://user-images.githubusercontent.com/112486797/192815777-f63b6981-08e6-42b1-8f4e-ea9732b6bb8d.png)


### New Sample Data Prediction
![image](https://user-images.githubusercontent.com/112486797/192815962-6364e277-ad89-423e-9062-d90943950e66.png)



## RESULT
Thus a convolutional deep neural network for digit classification and to verify the response for scanned handwritten images is written and executed successfully
