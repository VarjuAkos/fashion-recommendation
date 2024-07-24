# Fashion Recommendation System Assignment

## Objective:

Create a Fashion Recommendation System where users can upload a picture, and the system will recommend at least five or six similar pictures. The user interface should be built using Streamlit.

## Requirements:

### Part 1: Environment Setup
1. **Set up a Python Environment:**
    - Create a virtual environment using `venv` or `conda`.
    - Install necessary packages.

### Part 2: Model Creation
2. **Build the Feature Extraction Model:**
    - Use a pre-trained ResNet50 model from TensorFlow Keras to extract features from images.
    - Include the GlobalMaxPool2D layer to reduce the dimensionality of the feature vectors.

3. **Function to Extract Features:**
    - Create a function to load an image, preprocess it, and extract features using the model.

### Part 3: Recommendation System
4. **Prepare the Dataset:**
    - Collect a dataset of fashion images and save their paths. (Image Dataset is provided)
    - Extract and store features for each image in the dataset using the feature extraction function.

5. **Build the K-Nearest Neighbors (KNN) Model:**
    - Use `NearestNeighbors` from scikit-learn to find similar images.

6. **Find Similar Images:**
    - Create a function to find and return paths of similar images given an input image.

### Part 4: User Interface with Streamlit
7. **Create the Streamlit App:**
    - Design the user interface where users can upload an image and get recommendations.

## Deliverables:
1. **Python Code:**
    - The complete Python script with the model, feature extraction, recommendation logic, and Streamlit interface.
    - Comments and documentation explaining the code.

2. **Dataset:**
    - A folder containing the fashion images used for the recommendations.

3. **Streamlit App:**
    - A working Streamlit app that can be run locally to demonstrate the fashion recommendation system.

4. **ReadMe File:**
    - Instructions on how to set up and run the project.
    - Any assumptions made and additional notes.

## Evaluation Criteria:
- **Correctness:** The code runs without errors and provides correct recommendations.
- **Efficiency:** The recommendations are provided in a reasonable amount of time.
- **User Interface:** The Streamlit app is user-friendly and functions as expected.
- **Code Quality:** The code is well-structured, commented, and follows best practices.

**Note:** Interns can use their own logic to create the Fashion Recommendation System.