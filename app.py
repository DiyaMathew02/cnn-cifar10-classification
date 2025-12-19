import streamlit as st
import numpy as np
from PIL import Image
from tensorflow.keras.models import load_model

model = load_model("cifar10_cnn.h5")

classes = [
    "Airplane","Automobile","Bird","Cat","Deer",
    "Dog","Frog","Horse","Ship","Truck"
]

st.title("CIFAR-10 Image Classification")

file = st.file_uploader("Upload an image")

if file:
    image = Image.open(file).resize((32,32))
    img = np.array(image)/255.0
    img = img.reshape(1,32,32,3)

    prediction = model.predict(img)
    st.image(image)
    st.write("Prediction:", classes[np.argmax(prediction)])
