# Emotion Detector and Face Recognition

This project implements an Emotion Detector and Face Recognition system using Convolutional Neural Networks (CNNs) with TensorFlow and Keras. The system is capable of recognizing facial emotions and identifying faces of specific individuals.

## Emotion Detector

### Dataset

The emotion detection model is trained on a dataset containing images of facial expressions labeled with emotions. The dataset consists of the following emotion classes: Angry, Contempt, Disgusted, Fearful, Happy, Neutral, Sad, and Surprised.

### Preprocessing

The input images are preprocessed using the ImageDataGenerator class from Keras. The preprocessing steps include resizing the images to 48x48 pixels, converting them to grayscale, rescaling pixel values to be between 0 and 1, and applying data augmentation techniques such as random shifting and flipping.

### Model Architecture

The CNN architecture used for emotion detection consists of multiple convolutional layers followed by max-pooling layers to extract features from the input images. Fully connected layers with ReLU activation are added to classify the emotions. Dropout and batch normalization techniques are employed to prevent overfitting and improve model performance.

### Training and Evaluation

The model is trained using the training subset of the data and validated using the validation subset. The training progress is monitored using callbacks, and the best weights are saved. After training, the model's performance is evaluated by plotting the training and validation loss as well as the training and validation accuracy. A confusion matrix is also generated to assess the model's performance in classifying emotions.

## Face Recognition

### Dataset

The face recognition model is trained on a dataset containing images of individuals' faces. Each image is labeled with the corresponding person's identity. The dataset consists of images of Barack Obama, Bill Gates, and Melinda Gates.

### Preprocessing

Similar to the emotion detection model, the input images for face recognition are preprocessed using the ImageDataGenerator class from Keras. The images are resized to 48x48 pixels, converted to grayscale, and rescaled.

### Model Architecture

The CNN architecture used for face recognition follows a similar structure to the emotion detection model. It consists of convolutional layers, max-pooling layers, fully connected layers, dropout layers, and batch normalization layers. The final layer has softmax activation to classify the faces into different individuals.

### Training and Evaluation

The model is trained on the training subset of the face recognition dataset and validated on the validation subset. The training progress is monitored using callbacks, and the best weights are saved. The training and validation loss as well as the training and validation accuracy are plotted to assess the model's performance. A confusion matrix is generated to evaluate the model's ability to recognize faces accurately.

## Usage

To use the Emotion Detector and Face Recognition system:

1. Prepare the dataset: Ensure that you have properly organized datasets for both emotion detection and face recognition, containing labeled images of the desired emotions and individuals, respectively.

2. Install the required libraries: Make sure you have TensorFlow, Keras, OpenCV, and other necessary libraries installed. You can use the requirements.txt file to install the required packages.

3. Preprocess the data: Use the provided ImageDataGenerator to preprocess the data, including resizing, converting to grayscale, and applying data augmentation techniques.

4. Train the models: Build and train the emotion detection and face recognition models using the prepared datasets. Monitor the training progress and save the best weights using callbacks.

5. Evaluate the models: After training, evaluate the models' performance by plotting the loss and accuracy curves and generating confusion matrices.

6. Use the models for prediction: Use the trained models to predict emotions from facial expressions or

