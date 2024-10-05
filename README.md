# Brain_Stroke_Prediction_CNN
Project: Brain Stroke Prediction Using CNN

## Project Overview: 
This project aims to predict brain stroke conditions using Convolutional Neural Networks (CNNs). It leverages CT scan images to differentiate between normal brain conditions and stroke cases. By utilizing image processing and deep learning, the model seeks to effectively classify brain images to assist in the early detection of strokes.

## Key Sections and Steps:

### 1.Importing Libraries: 
The project imports several essential libraries, such as TensorFlow, Keras, OpenCV, and others for tasks like image preprocessing, model creation, and evaluation. This step sets up the environment needed for the development of a CNN-based model.

Code: 
import warnings
warnings.filterwarnings('ignore')

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.metrics import classification_report , confusion_matrix , accuracy_score
from sklearn.model_selection import train_test_split

import cv2
#from google.colab.patches import cv2_imshow
from PIL import Image 
import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Input, Dense,Conv2D , MaxPooling2D, Flatten,BatchNormalization,Dropout
from tensorflow.keras.preprocessing import image_dataset_from_directory
import tensorflow_hub as hub 

### 2.Image Data Preprocessing:
The dataset is organized into two categories: "Normal" and "Stroke" brain images.
The paths to these categories are defined, and the image files are accessed.
This step is crucial to prepare the image data before feeding it into the model, ensuring that the images are in a compatible format for training.

Code:
normal_path = "/kaggle/input/brain-stroke-ct-image-dataset/Brain_Data_Organised/Normal"
stroke_path = "/kaggle/input/brain-stroke-ct-image-dataset/Brain_Data_Organised/Stroke"
normal_folder = os.listdir(normal_path)
stroke_folder = os.listdir(stroke_path)
data = []

for img_file in normal_folder:
    image = Image.open("/kaggle/input/brain-stroke-ct-image-dataset/Brain_Data_Organised/Normal/" + img_file)
    image = image.resize((224,224))
    image = image.convert('RGB')
    image = np.array(image)
    data.append(image)
    
for img_file in stroke_folder:
    image = Image.open("/kaggle/input/brain-stroke-ct-image-dataset/Brain_Data_Organised/Stroke/" + img_file)
    image = image.resize((224,224))
    image = image.convert('RGB')
    image = np.array(image)
    data.append(image)
normal_label = [0]*len(normal_folder)
stroke_label = [1]*len(stroke_folder)
Target_label = normal_label + stroke_label
x = np.array(data)
y = np.array(Target_label)
x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.10,shuffle=True)
x_train_s = x_train/255
x_test_s = x_test/255

### 3.Deep Learning Model Architecture:
Although not yet visible in the first few cells, it is likely that the next steps involve constructing a CNN model. The Keras library is used to define layers such as convolutional layers, pooling layers, and fully connected layers.
The model will be trained to recognize features in brain CT images that distinguish between normal and stroke-affected patients.

Code:
model = Sequential()

model.add(Conv2D(32,kernel_size=(3,3),strides=1,padding="valid",activation="relu", input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(64,kernel_size=(3,3),strides=1,padding="valid",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Conv2D(128,kernel_size=(3,3),strides=1,padding="valid",activation="relu"))
model.add(MaxPooling2D(pool_size=2))

model.add(Flatten())
model.add(Dense(256, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.2))
model.add(Dense(1, activation="sigmoid"))

model.compile(optimizer="adam",loss="binary_crossentropy",
             metrics=["accuracy"])

model.summary()

### 4.Training and Evaluation:
The model will be trained on the prepared image dataset. Performance metrics such as accuracy, classification report, and confusion matrix will be used to evaluate its performance.
These metrics help determine the model's ability to classify brain images correctly and guide any necessary adjustments for improvement.

Code:
loss, acc = model.evaluate(x_test_s,y_test)
print("Loss on Test Set:",loss)
print("Accuracy on Test Set:",acc)

loss, acc = model.evaluate(x_train_s,y_train)
print("Loss on Train Set:",loss)
print("Accuracy on Train Set:",acc)
8/8 ━━━━━━━━━━━━━━━━━━━━ 0s 22ms/step - accuracy: 1.0000 - loss: 0.0058  
Loss on Test Set: 0.00431433180347085
Accuracy on Test Set: 1.0
71/71 ━━━━━━━━━━━━━━━━━━━━ 1s 18ms/step - accuracy: 1.0000 - loss: 5.2578e-04
Loss on Train Set: 0.00050237902905792
Accuracy on Train Set: 1.0
y_pred_test = model.predict(x_test_s)
y_pred_test_label = [1 if i>=0.5 else 0 for i in y_pred_test]
print("-----Metrics Evaluation On Test Set -----")
print()
print("Confusion Matrix:\n",confusion_matrix(y_test,y_pred_test_label))
print()
print("Classification Report:\n",classification_report(y_test,y_pred_test_label))
-----Metrics Evaluation On Test Set -----

Confusion Matrix:
 [[165   0]
 [  0  86]]

Classification Report:
               precision    recall  f1-score   support

           0       1.00      1.00      1.00       165
           1       1.00      1.00      1.00        86

    accuracy                           1.00       251
   macro avg       1.00      1.00      1.00       251
weighted avg       1.00      1.00      1.00       251

### 5.Expected Outcome:
The project aims to build a predictive model capable of accurately identifying stroke conditions from CT images.
The output can potentially assist healthcare professionals by providing an early warning for stroke conditions, enabling prompt intervention.
