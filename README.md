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

The dataset comprises a total of 43,390 cell images captured through blood smear tests. It is divided into two classes: Parasitised and Uninfected. The dataset contains training and test sets. Additionally, to establish a validation set, 20% of the training data has been allocated.
