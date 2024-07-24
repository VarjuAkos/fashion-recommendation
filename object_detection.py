import tensorflow as tf
import numpy as np
from PIL import Image

# Load a pre-trained model (for this example, we'll use MobileNetV2)
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
    
    # Filter for clothing items
    clothing_items = [
        'shirt', 'jersey', 'tee_shirt', 'jean', 'trouser', 'dress', 
        'skirt', 'suit', 'jacket', 'coat', 'sweater', 'shoe', 'sneaker', 'maillot',
    ]

    print(decoded_predictions)
    
    detected_items = [
        item[1] for item in decoded_predictions[0] 
        if any(clothing in item[1] for clothing in clothing_items)
    ]
    
    return list(set(detected_items))  # Remove duplicates

# Test the function
if __name__ == "__main__":
    test_image = Image.open('./set.jpg')
    detected = detect_objects(test_image)
    print("Detected items:", detected)