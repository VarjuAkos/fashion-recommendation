# Fashion Recommendation System: High-Level Design

### Key components:

1. **User Interface**: Built using Streamlit, providing an intuitive way for users to interact with the system.
2. **Object Detection**: Utilizes MobileNetV2 to identify fashion items in uploaded images.
3. **Image Processing**: Includes an interactive cropping tool for users to select specific areas of interest.
4. **Feature Extraction**: Employs ResNet50 to extract meaningful features from the selected image area.
5. **Recommendation Engine**: Uses K-Nearest Neighbors algorithm to find similar fashion items based on extracted features.
6. **Data Management**: Handles the storage and retrieval of fashion item features and image paths.

### Workflow:
1. User uploads an image
2. Image is displayed and passed to object detection
3. Detected items are presented in a dropdown
4. User crops the image using an interactive tool
5. Cropped image is passed to feature extraction
6. Extracted features are used to find recommendations
7. Recommendations are displayed to the user


### Output:

The system provides the following outputs:

1. Detected fashion items in the uploaded image, displayed in a dropdown menu.
2. An interactive cropping interface for the user to select a specific area of interest.
3. A display of the cropped area selected by the user.
4. A set of 5 recommended fashion items similar to the selected area, displayed as images.