# CNN Image Classification using CIFAR-10

## Project Overview

This project focuses on designing and implementing a **Convolutional Neural Network (CNN)** for **image classification** using the **CIFAR-10 dataset**. The objective is to classify input images into one of ten predefined categories such as airplanes, automobiles, birds, cats, deer, dogs, frogs, horses, ships, and trucks.

The project demonstrates the complete machine learning workflow including data preprocessing, CNN model construction, training, evaluation, visualization of results, and deployment through a simple user interface.

## Dataset

**CIFAR-10 Dataset**

* 60,000 RGB images (32 × 32 pixels)
* 10 image classes
* 50,000 training images
* 10,000 testing images

The dataset is loaded using `tensorflow.keras.datasets`.

## Project Structure

```
cnn-cifar10-classification/
│
├── app.py               # Streamlit app for image prediction
├── train.py             # Model training and visualization
├── evaluate.py          # Model evaluation
├── preprocess.py        # Dataset preprocessing
├── cifar10_cnn.h5       # Saved trained CNN model
├── accuracy.png         # Accuracy graph
├── loss.png             # Loss graph
├── requirements.txt     # Required Python packages
├── README.md            # Project documentation
└── .gitignore
```

## CNN Concepts Demonstrated

* Convolution layers for feature extraction
* ReLU activation function
* Max pooling for spatial reduction
* Fully connected (Dense) layers
* Softmax output layer for classification
* Adam optimizer
* Categorical Cross-Entropy loss function

## Model Training

* The CNN model is trained for **10 epochs**
* Training and validation accuracy and loss are recorded
* The trained model is saved as **cifar10_cnn.h5**

## Result Visualization

The following plots are generated during training:

* **accuracy.png** – Training vs Validation Accuracy
* **loss.png** – Training vs Validation Loss

These visualizations help in analyzing model performance.

## User Interface

A simple **Streamlit-based web interface** is included:

* Users can upload an image
* The model predicts and displays the image class
* Provides an interactive experience for testing the CNN

## How to Run the Project

### Step 1: Install dependencies

```bash
pip install -r requirements.txt
```

### Step 2: Train the model

```bash
python train.py
```

### Step 3: Evaluate the model

```bash
python evaluate.py
```

### Step 4: Run the Streamlit application

```bash
streamlit run app.py
```

## Deliverables

* Python source code
* Trained CNN model
* Performance visualizations
* GitHub repository with documentation

## Conclusion

This project successfully implements a CNN-based image classification system using CIFAR-10. It clearly demonstrates core CNN concepts such as convolution, pooling, activation functions, optimization, and evaluation. The project also includes visual outputs and a user interface, making it suitable for academic submission.

## Author

**Diya Mathew**