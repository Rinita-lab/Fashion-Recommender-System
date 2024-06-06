# Fashion-Recommender-System
This project implements a deep learning-based fashion recommender system using the ResNet50 architecture. The recommender system suggests similar fashion items based on image content. This README provides an overview of the project which groups visually similar items.

# Project Overview
The fashion recommender system leverages the ResNet50 model to extract features from fashion images. These features are then used to recommend similar items through content-based filtering. The system can be extended to incorporate collaborative filtering using user interaction data.

# Prerequisites
Python 3.10
TensorFlow 2.16.1
Keras
NumPy
Scikit-learn

# Workflow
The development of the deep learning-based fashion recommender system using a Convolutional Neural Network (CNN) with transfer learning from a pre-trained ResNet50 model involves several systematic steps:
# Data Collection and Preparation: 
Gather a comprehensive dataset of fashion images from sources such as DeepFashion or Fashion-MNIST. Organize the dataset into training, validation, and test directories.
# Data Preprocessing: 
Utilize Keras' ImageDataGenerator for image preprocessing, including resizing images to 224x224 pixels and normalizing pixel values. Create data generators for loading the training, validation, and test datasets.
# Transfer Learning with ResNet50: 
Import the pre-trained ResNet50 model without the top layer. This step leverages transfer learning to utilize the feature extraction capabilities of ResNet50, which has been trained on a large image dataset (ImageNet).
# Feature Extraction: 
Add a global average pooling layer to the ResNet50 model to generate fixed-size feature vectors for each image. Use this modified model to extract high-level features from the images in the dataset. This transforms the raw image data into feature vectors that capture essential visual patterns.
# Applying K-Nearest Neighbors (KNN): 
Use the extracted feature vectors to build a K-Nearest Neighbors model. Calculate the Euclidean distance between feature vectors to determine the similarity between images.
# Recommending 5 Best Results: 
For a given input image, use the KNN model to identify the 5 most similar fashion items based on the smallest Euclidean distances. These top 5 recommendations represent the best matches in the dataset.

This workflow ensures a systematic approach to developing a robust and efficient fashion recommender system, combining the strengths of deep learning, transfer learning, and traditional KNN algorithms to deliver high-quality recommendations.
