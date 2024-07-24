import tensorflow as tf
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.layers import GlobalMaxPooling2D
import numpy as np

class FeatureExtractor:
    def __init__(self):
        # Load pre-trained ResNet50 model without top layers
        base_model = ResNet50(weights='imagenet', include_top=False)
        
        # Add a Global Max Pooling layer
        x = base_model.output
        x = GlobalMaxPooling2D()(x)
        
        self.model = tf.keras.Model(inputs=base_model.input, outputs=x)

    def extract_features(self, img):
        # Resize and preprocess the image
        img = img.resize((224, 224))
        img_array = tf.keras.preprocessing.image.img_to_array(img)
        expanded_img_array = np.expand_dims(img_array, axis=0)
        preprocessed_img = preprocess_input(expanded_img_array)

        # Extract features
        features = self.model.predict(preprocessed_img)
        flattened_features = features.flatten()
        normalized_features = flattened_features / np.linalg.norm(flattened_features)
        
        return normalized_features

# Test the feature extractor
if __name__ == "__main__":
    from PIL import Image
    
    # Create an instance of FeatureExtractor
    feature_extractor = FeatureExtractor()
    
    # Load a test image
    test_image = Image.open("path_to_test_image.jpg")
    
    # Extract features
    features = feature_extractor.extract_features(test_image)
    
    print("Feature vector shape:", features.shape)
    print("Sample of feature vector:", features[:10])  # Print first 10 features