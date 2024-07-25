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

def detect_objects_with_boxes(image):
    preprocessed_image = preprocess_image(image)
    predictions = model.predict(preprocessed_image)
    decoded_predictions = tf.keras.applications.mobilenet_v2.decode_predictions(predictions)
    
    # Filter for clothing items
    clothing_items = [
        'shirt', 'jersey', 'tee_shirt', 'jean', 'trouser', 'dress', 
        'skirt', 'suit', 'jacket', 'coat', 'sweater', 'shoe', 'sneaker'
    ]
    
    detected_items = []
    for item in decoded_predictions[0]:
        if any(clothing in item[1] for clothing in clothing_items):
            # For simplicity, we're using the whole image as the bounding box
            # In a real scenario, you'd use an object detection model to get accurate boxes
            box = (0, 0, image.width, image.height)
            detected_items.append((item[1], box))
    
    return detected_items

# Test the function
if __name__ == "__main__":
    test_image = Image.open("path_to_test_image.jpg")
    detected = detect_objects_with_boxes(test_image)
    print("Detected items:", detected)