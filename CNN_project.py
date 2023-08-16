#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image
import matplotlib.pyplot as plt


# In[2]:


# Data augmentation configuration for training data
train_datagen = ImageDataGenerator(
    rescale=1./255,             # Rescale pixel values to [0, 1]
    rotation_range=30,          # Rotate images randomly by up to 30 degrees
    width_shift_range=0.2,      # Shift images horizontally by up to 20% of the width
    height_shift_range=0.2,     # Shift images vertically by up to 20% of the height
    shear_range=0.2,            # Apply shear transformations
    zoom_range=0.3,             # Apply zoom transformations
    horizontal_flip=True,       # Flip images horizontally
    fill_mode='nearest',        # Fill new pixels with the nearest available pixel
    brightness_range=(0.5, 1.5),  # Adjust brightness of images
    validation_split=0.2        # Specify the validation split here
)

#Data preprocessing
test_datagen = ImageDataGenerator(
    rescale=1./255             # Rescale pixel values to [0, 1]
)

# Generating training data generator
training_set = train_datagen.flow_from_directory(
    'MalariaCells/training_set',  # Path to the training dataset
    target_size=(64, 64),      # Set the target size (width, height) for resizing
    batch_size=220,            # Number of images in each batch
    class_mode='binary',       # For binary classification
    subset='training'          # Subset of the data (training portion)
)

# Generating validation data generator
validation_set = train_datagen.flow_from_directory(
    'MalariaCells/training_set',  # Path to the training dataset
    target_size=(64, 64),      # Set the target size (width, height) for resizing
    batch_size=50,             # Number of images in each batch
    class_mode='binary',       # For binary classification
    subset='validation'        # Subset of the data (validation portion)
)

# Generating testing data generator
test_set = test_datagen.flow_from_directory(
    'MalariaCells/testing_set',  # Path to the testing dataset
    target_size=(64, 64),      # Set the target size (width, height) for resizing
    batch_size=150,            # Number of images in each batch
    class_mode='binary'        # For binary classification
)


# In[3]:


# Create a Sequential model
model = Sequential([
    # First Convolutional layer
    Conv2D(filters=16, kernel_size=(4, 4), input_shape=(64, 64, 3)),
    BatchNormalization(),  # Add BatchNormalization after Conv2D
    Activation('relu'),    # Add Activation layer
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling
    
    # Second Convolutional layer
    Conv2D(filters=32, kernel_size=(4, 4), kernel_regularizer=l2(0.01)),
    BatchNormalization(),  # Add BatchNormalization after Conv2D
    Activation('relu'),    # Add Activation layer
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling
    
    # Third Convolutional layer
    Conv2D(filters=64, kernel_size=(4, 4), kernel_regularizer=l2(0.01)),
    BatchNormalization(),  # Add BatchNormalization after Conv2D
    Activation('relu'),    # Add Activation layer
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling
    
    # Fourth Convolutional layer
    Conv2D(filters=128, kernel_size=(4, 4), kernel_regularizer=l2(0.01)),
    BatchNormalization(),  # Add BatchNormalization after Conv2D
    Activation('relu'),    # Add Activation layer
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling
    
    Flatten(),  # Flatten feature maps to 1D
    Dense(32, activation = "relu"), # Fully connected layer with 32 neurons and ReLU activation
    Dense(16, activation = "relu"), # Fully connected layer with 16 neurons and ReLU activation
    Dense(1, activation='sigmoid')  # Fully connected layer for binary classification
])


# In[4]:


model.summary()


# In[5]:


# Compiling the model with Adam optimizer and metrics including accuracy and precision
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')])

# Define a callback to reduce the learning rate when validation loss plateaus
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=2, verbose=1)

# Train the model using the training set and validate using the validation set
history = model.fit(training_set, epochs=20, validation_data=validation_set, callbacks=[reduce_lr])


# In[6]:


# Plot training & validation accuracy values
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
#Adding title and labels to x and y axis
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()

# Plot training & validation loss values
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
#Adding title and labels to x and y axis
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Train', 'Validation'], loc='lower right')
plt.show()


# In[7]:


loss,accuracy,precision,recall,auc = model.evaluate(test_set)


# In[8]:


print(f"Test accuracy: {accuracy:.4f}")
print(f"Test precision: {precision:.4f}")
print(f"Test recall: {recall:.4f}")
print(f"Test AUC: {auc:.4f}")


# In[9]:


model.save('model.h5')


# In[ ]:




