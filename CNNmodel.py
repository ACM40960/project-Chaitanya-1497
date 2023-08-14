#Importing libraries

from tensorflow.keras.models import Sequential
from tensorflow.keras.metrics import Precision, Recall, AUC
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Activation
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, TensorBoard, CSVLogger
from tensorflow.keras.regularizers import l1, l2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# Data Preprocessing
train_datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)  # Normalize pixel values to [0, 1]
test_datagen = ImageDataGenerator(rescale=1./255)

training_set = train_datagen.flow_from_directory('Copy training set file location here',
                                                target_size=(64, 64),  # Set the target size (width, height) for resizing
                                                batch_size=220,
                                                class_mode='binary',
                                                subset='training')  # For binary classification. Use 'categorical' for multi-class

validation_set = train_datagen.flow_from_directory('Copy training set file location here',
                                                target_size=(64, 64),  
                                                batch_size=50,
                                                class_mode='binary',
                                                subset='validation')  


test_set = test_datagen.flow_from_directory('Copy test set file location here',
                                            target_size=(64, 64),
                                            batch_size=150,
                                            class_mode='binary')

# Create a Sequential model
model = Sequential([
    # First Convolutional layer
    Conv2D(filters=8, kernel_size=(4, 4), input_shape=(64, 64, 3)),
    BatchNormalization(),  # Add BatchNormalization after Conv2D
    Activation('relu'),    # Add Activation layer
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling
    
    # Second Convolutional layer
    Conv2D(filters=8, kernel_size=(4, 4), kernel_regularizer=l2(0.01)),
    BatchNormalization(),  # Add BatchNormalization after Conv2D
    Activation('relu'),    # Add Activation layer
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling
    
    # Third Convolutional layer
    Conv2D(filters=16, kernel_size=(4, 4), kernel_regularizer=l2(0.01)),
    BatchNormalization(),  # Add BatchNormalization after Conv2D
    Activation('relu'),    # Add Activation layer
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling
    
    # Fourth Convolutional layer
    Conv2D(filters=16, kernel_size=(4, 4), kernel_regularizer=l2(0.01)),
    BatchNormalization(),  # Add BatchNormalization after Conv2D
    Activation('relu'),    # Add Activation layer
    MaxPooling2D(pool_size=(2, 2)),  # Max pooling
    
    Flatten(),  # Flatten the 3D feature maps to 1D
    Dense(1, activation='sigmoid'),  # Fully connected layer for binary classification
    Dropout(0.1)  # Dropout layer to reduce overfitting
])

model.summary()

# Compiling the model with Adam optimizer and metrics including accuracy and precision
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy', Precision(name='precision'), Recall(name='recall'), AUC(name='auc')])

# Define a callback to reduce the learning rate when validation loss plateaus
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, verbose=1)

# Train the model using the training set and validate using the validation set
model.fit(training_set, epochs=10, validation_data=validation_set, callbacks=[reduce_lr])

#Extracting the metrics
loss,accuracy,precision,recall,auc = model.evaluate(test_set)

print(f"Test accuracy: {accuracy:.4f}")
print(f"Test precision: {precision:.4f}")
print(f"Test recall: {recall:.4f}")
print(f"Test AUC: {auc:.4f}")

#Saving the model
model.save('model.h5')
