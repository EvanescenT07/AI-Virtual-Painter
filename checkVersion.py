from PIL import Image 

import pytesseract 

import keras
import tensorflow as tf

# Get Keras and TensorFlow versions
keras_version = keras.__version__
tensorflow_version = tf.__version__

keras_version, tensorflow_version

# Get pytesseract version
pytesseract_version = pytesseract.get_tesseract_version()

pytesseract_version

print(keras_version)
print(tensorflow_version)
print(pytesseract_version)