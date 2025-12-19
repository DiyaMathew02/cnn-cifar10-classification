from tensorflow.keras.models import load_model
from preprocess import load_and_preprocess

_, x_test, _, y_test = load_and_preprocess()

model = load_model("cifar10_cnn.h5")

loss, accuracy = model.evaluate(x_test, y_test)
print("Accuracy:", accuracy)
