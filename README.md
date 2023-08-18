# Malaria Diagnosis (Disease Modeling)

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

<img src="https://github.com/ACM40960/project-Chaitanya-1497/assets/133139835/ebd0e6d7-c961-474a-b40b-3913f255d483" alt="CNN Layers" width="600"/>

*An example of a CNN model*

The model consists of four convolutional layers, each using Rectified Linear Unit (ReLU) activation functions to introduce nonlinearity. The input data has a shape of (64, 64, 3). The convolutional layers employ ascending numbers of filters: 16, 32, 64, and 128 respectively. To accelerate training and serve as a regularizer, the Batch Normalization technique is utilized. Additionally, L2 regularization is employed to enhance generalization and mitigate overfitting. After the convolutional layers, the data is flattened and passed through a Deep Neural Network (DNN). The initial two layers of the DNN facilitate the learning of intricate data relationships. Finally, the output layer employs a sigmoid activation function to achieve binary classification.

![model_image](https://github.com/ACM40960/project-Chaitanya-1497/assets/133139835/4685ce92-c0b4-4f78-9c0d-4322e9373629)

The model runs for 20 epochs with a batch size of 101.

## Results

After training our Convolutional Neural Network (CNN) model on the Malaria dataset, we evaluated its performance on the test set using various metrics. The evaluation results are as follows:

**Accuracy:** The accuracy of 0.9563 implies that our model correctly classified approximately 95.63% of the test samples, showcasing its ability to make accurate predictions.

**Precision:** With a precision of 0.9660, the model demonstrates its proficiency in minimizing false positive errors by accurately identifying true positive cases.

**Recall:** The recall value of 0.9463 signifies that the model effectively captured approximately 94.63% of the actual positive cases in the dataset, highlighting its sensitivity to detecting infected instances.

**AUC (Area Under the Curve):** The AUC value of 0.9889 corresponds to the area under the Receiver Operating Characteristic (ROC) curve, indicating a strong ability of the model to distinguish between the positive and negative classes.

## Web Application
The web application has been developed using Flask, a web framework that enables developers to create lightweight web applications rapidly and efficiently through Flask Libraries. To explore the application's functionality, ensure that the provided static, templates, app.py, and model.h5 files from this repository are placed within the same folder. Below, you will find images that illustrate the functioning of the application:

<img src="https://github.com/ACM40960/project-Chaitanya-1497/assets/133139835/8eb9485d-8cae-4154-9455-42498b9f6035" alt="MalariaNeg" width="600"/>
<img src="https://github.com/ACM40960/project-Chaitanya-1497/assets/133139835/15deb00c-ae63-425e-82e4-f8a1cdc20e29" alt="MalariaP" width="600"/>

This application has been created to assist lab technicians. It empowers them to upload cell images acquired from microscopes. Upon submitting the image, the application swiftly generates predictions using the model.

## Conclusion
Our CNN model demonstrates a commendable level of sensitivity and precision in Malaria diagnosis. This advancement has the potential to expedite decision-making for lab technicians. As we look ahead, refining and expanding the model holds the promise of even more promising outcomes.

## References
1. S. V. Militante, ”Malaria Disease Recognition through Adaptive Deep Learning Models of Convolutional Neural Network,” 2019 IEEE 6th International Conference on Engineering Technologies and Applied Sciences (ICETAS), Kuala Lumpur, Malaysia, 2019, pp.1-6, doi: 10.1109/ICETAS48360.2019.9117446.

2. https://lhncbc.nlm.nih.gov/LHC-research/LHC-projects/image-processing/malaria-screener.html

3. https://www.cdc.gov/
   
4. Khongdet Phasinam, Tamal Mondal, Dony Novaliendry,Cheng-Hong Yang, Chiranjit Dutta, Mohammad Shabaz,”Analyzing the Performance of Machine Learning Techniques in
Disease Prediction”, Journal of Food Quality, vol. 2022, Article ID 7529472, 9 pages, 2022

## Authors

- Chaitanya Ashok Malagikar
- Vedanti Mahesh Borate
