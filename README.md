# Malaria Diagnosis (Disease Modeling)
Repository for Malaria Diagnosis

## Introduction:
Malaria is a deadly disease caused by parasites transmitted to humans through the bite of an infected mosquito. It remains a significant public health challenge in various parts of the world, affecting approximately 200 million people each year, with around 400,000 lives lost.
The project aims to use cell images acquired using blood smear tests to build a predictive model for Malaria diagnosis. This model can work in conjunction with traditional microscopy methods to expedite the process of obtaining diagnostic results.

### Diagnosis Methods

1. Traditional Method : Microscope Diagnosis - Malaria is diagnosed generally by microscopic examination of blood. However, this method 
   can be time-consuming and requires a trained technician.

2. Rapid Diagnostic Test - RDTs work by detecting the presence of malaria antigens in a blood sample.
   
3. Recently, as machine learning has enhanced, there has been a growing interest in automating diagnosis of malaria. One of the approaches to automated malaria diagnosis is using convolutional neural network (CNN) model. CNNs models are specially designed for image recognition. They have been effective at classifying malaria cells in images.

## Implementation
The entire model has been developed within a Jupyter Notebook environment. For running the model, a Python application is required. You can utilize Anaconda to install Jupyter Notebook, which includes a selection of pre-installed packages. Additionally, to execute the web application files(.py), integrated development environments (IDEs) like PyCharm and Spyder can be employed. These IDEs are also conveniently accessible through the Anaconda platform.

## Dataset
The dataset we hve used is taken from Kaggle.

Link: "https://www.kaggle.com/datasets/miracle9to9/files1"

The dataset comprises a total of 43,390 cell images captured through blood smear tests. It is divided into two classes: Parasitised and Uninfected. The dataset contains training and test sets. Additionally, to establish a validation set, 20% of the training data has been allocated. Data augmentation is used in our model to increase the robustness.

## Libraries Used
```
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.regularizers import  l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt
```

The following are the libraries used to run the web application in the .py file
```
from flask import Flask, render_template, request
from keras.models import load_model
from keras.preprocessing import image
```
## Model
The model consists of four convolutional layers, each using Rectified Linear Unit (ReLU) activation functions to introduce nonlinearity. The input data has a shape of (64, 64, 3). The convolutional layers employ ascending numbers of filters: 16, 32, 64, and 128 respectively. To accelerate training and serve as a regularizer, the Batch Normalization technique is utilized. Additionally, L2 regularization is employed to enhance generalization and mitigate overfitting. After the convolutional layers, the data is flattened and passed through a Deep Neural Network (DNN). The initial two layers of the DNN facilitate the learning of intricate data relationships. Finally, the output layer employs a sigmoid activation function to achieve binary classification.

![model_image](https://github.com/ACM40960/project-Chaitanya-1497/assets/133139835/4685ce92-c0b4-4f78-9c0d-4322e9373629)


```
