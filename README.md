# CNN CIFAR-10 Image Classification

## Project Overview
This project implements a **Convolutional Neural Network (CNN)** for image classification using the **CIFAR-10 dataset**.  
The goal of this project is to design, train, and evaluate a CNN model that can correctly classify images into one of 10 predefined categories.

A **Streamlit web application** is also developed as part of this project, allowing users to upload an image and view the model’s prediction.

## Dataset Used
- **CIFAR-10 Dataset**
- Total images: 60,000
- Image size: 32 × 32 (RGB)
- Number of classes: 10

### CIFAR-10 Classes
- Airplane  
- Automobile  
- Bird  
- Cat  
- Deer  
- Dog  
- Frog  
- Horse  
- Ship  
- Truck  

## CNN Concepts Covered
- Convolutional layers for feature extraction
- Max pooling for dimensionality reduction
- ReLU activation function
- Fully connected (Dense) layers
- Softmax output layer for multi-class classification
- Adam optimizer
- Categorical Cross-Entropy loss function

## Project Structure
```
cnn-cifar10-classification/
│
├── preprocess.py    # Loads and preprocesses CIFAR-10 dataset
├── train.py         # Builds and trains the CNN model
├── evaluate.py      # Evaluates trained model performance
├── app.py           # Streamlit web application for prediction
├── requirements.txt # List of required Python libraries
├── README.md        # Project documentation
```

## Requirements
The following Python libraries are required:

- TensorFlow
- NumPy
- Streamlit
- Pillow

Install all dependencies using:
```
pip install -r requirements.txt
```

## How to Run the Project

### Step 1: Train the Model
Run the following command to train the CNN:
```
python train.py
```
This will train the model and save it as `cifar10_cnn.h5`.

### Step 2: Evaluate the Model
Evaluate the trained model using:
```
python evaluate.py
```
This will display the accuracy and performance metrics of the model.

### Step 3: Run the Streamlit Application
Launch the user interface using:
```
streamlit run app.py
```
Upload an image and view the predicted CIFAR-10 class.

## Output
- The trained CNN model accurately classifies CIFAR-10 images.
- The Streamlit app provides an interactive interface for image upload and prediction.

## Result Visualizations
- Training and validation accuracy graph
- Training and validation loss graph

## Conclusion
This project demonstrates the complete workflow of a CNN-based image classification system, including data preprocessing, model training, evaluation, and deployment through a web interface.
