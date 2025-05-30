# Deepfake-Detector
Deepfake detector for Real vs AI generated images

# Real vs. AI-Generated Faces Classification

This repository contains the code for a deep learning model that classifies images as either real or AI-generated faces. The project utilizes a Convolutional Neural Network (CNN) built with TensorFlow and Keras.

## Project Overview

The goal of this project is to develop a model capable of distinguishing between authentic human faces and those created by artificial intelligence. This is a growing area of interest due to the increasing sophistication of generative models.

## Dataset

The project uses the "[Real vs. AI-Generated Faces Dataset](https://www.kaggle.com/datasets/philosopher0808/real-vs-ai-generated-faces-dataset)" from Kaggle. This dataset contains a collection of images labeled as either real or AI-generated.

## Code Description

The provided notebook performs the following steps:

1.  **Data Loading and Exploration:** Imports the dataset from Kaggle, explores the directory structure, and visualizes the class distribution.
2.  **Data Preprocessing and Augmentation:**
    *   A `DataGenerator` class is used to load and prepare the image data in batches.
    *   Data augmentation techniques (random horizontal flip, contrast, and brightness adjustments) are applied to the training set to improve model generalization.
3.  **Exploratory Data Analysis (EDA):**
    *   Sample images from the training, validation, and test sets are visualized.
    *   Principal Component Analysis (PCA) is performed on a subset of the training data to understand the dimensionality and variance of the image features.
    *   A boxplot of the first few principal components is generated to identify potential outliers.
    *   A dummy hypothesis test (independent samples t-test) is included as an example of how statistical tests could be used to compare features between the two classes (though a real application would require extracting meaningful features first).
4.  **Model Architecture:** A sequential CNN model is defined with convolutional, pooling, flatten, and dense layers. Dropout is included for regularization.
5.  **Model Compilation:** The model is compiled with the Adam optimizer, categorical crossentropy loss, and accuracy as the evaluation metric.
6.  **Model Training:** The model is trained on the augmented training data, with validation performed on the validation set. Early stopping and model checkpointing callbacks are used.
7.  **Model Evaluation:** The trained model is evaluated on the test set to assess its performance.
8.  **Results Analysis:** A classification report and confusion matrix are generated to provide detailed performance metrics. Training and validation accuracy and loss curves are plotted.

## Requirements

To run this code, you will need:

*   Python 3.x
*   TensorFlow 2.x
*   Keras
*   NumPy
*   Matplotlib
*   Seaborn
*   Scikit-learn
*   OpenCV (cv2)
*   KaggleHub (for dataset download)

You can install the required libraries using pip:
Use code with caution
bash pip install tensorflow matplotlib seaborn scikit-learn opencv-python kagglehub

## Usage

1.  **Clone the repository:**
Use code with caution
bash git clone cd

2.  **Download the dataset:** The notebook includes code to download the dataset from Kaggle using `kagglehub`. Make sure you have the necessary Kaggle API credentials configured.
3.  **Run the Jupyter notebook:** Open the notebook in a Jupyter environment (like Google Colab or a local Jupyter installation) and run the cells sequentially.

## Results

The notebook outputs the following evaluation metrics:

*   Test Loss
*   Test Accuracy
*   Classification Report (Precision, Recall, F1-score for each class)
*   Confusion Matrix
*   Plots of training and validation accuracy and loss over epochs.

## Future Improvements

*   Experiment with different CNN architectures (e.g., VGG, ResNet).
*   Implement more advanced data augmentation techniques.
*   Explore transfer learning by fine-tuning a pre-trained model.
*   Investigate techniques for handling potential biases in the dataset.
*   Develop a user interface or API for classifying new images.

## License

[Specify your chosen license here, e.g., MIT License]

## Acknowledgements

*   The creators of the "Real vs. AI-Generated Faces Dataset" on Kaggle.
*   The developers of TensorFlow, Keras, and other libraries used in this project.
