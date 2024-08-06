import tensorflow as tf
import numpy as np
from PIL import Image

# Load a pre-trained model (MobileNetV2)
model = tf.keras.applications.MobileNetV2(weights='imagenet')

def preprocess_image(image):
    image = image.resize((224, 224))
    image = tf.keras.preprocessing.image.img_to_array(image)
    image = tf.keras.applications.mobilenet_v2.preprocess_input(image)
    image = np.expand_dims(image, axis=0)
    return image

def detect_objects(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)
    print("--Detecting objects--")
    # Return top 5 predictions
    return decoded_predictions[0][:5]
