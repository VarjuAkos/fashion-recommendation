# Fashion Recommendation System: High-Level Design

The project consists of several key components:

1. **User Interface**: Built using Streamlit, providing an intuitive way for users to interact with the system.
2. **Object Detection**: Utilizes MobileNetV2 to identify fashion items in uploaded images.
3. **Image Processing**: Includes an interactive cropping tool for users to select specific areas of interest.
4. **Feature Extraction**: Employs ResNet50 to extract meaningful features from the selected image area.
5. **Recommendation Engine**: Uses K-Nearest Neighbors algorithm to find similar fashion items based on extracted features.
6. **Data Management**: Handles the storage and retrieval of fashion item features and image paths.
