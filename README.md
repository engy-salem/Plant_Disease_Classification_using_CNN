# 🌿 Plant Disease Classification using CNN

A Deep Learning project that classifies plant leaf images into 38 categories using a Convolutional Neural Network (CNN) built with PyTorch. The project includes a web interface built with Gradio and deployed on Hugging Face Spaces.

## 🚀 Live Demo

Try the application here:

**[🌱 Plant Disease Classification — Live Demo](https://huggingface.co/spaces/EngySalem/plant-disease-classification)**

Upload a plant leaf image to get the model's top 3 predicted classes and their confidence scores.

## 📌 Project Overview

Plant diseases can affect crop quality and agricultural productivity. This project uses deep learning and image classification to identify plant diseases from leaf images.

The trained CNN model predicts the most likely plant disease category based on an uploaded image. A Gradio interface makes the model accessible through a web browser without requiring users to run the code locally.

## 📊 Dataset

- **Dataset:** PlantVillage Full
- **Total Images:** 54,304
- **Number of Classes:** 38
- **Dataset Source:** [PlantVillage Full on Hugging Face](https://huggingface.co/datasets/geraldmc/plantvillage-full)

The dataset contains plant leaf images representing different plant species and disease categories.

## 🧠 Model Architecture

The project uses a custom Convolutional Neural Network (CNN) implemented with PyTorch.

The architecture includes:

- Convolutional layers for extracting visual features.
- ReLU activation functions.
- Max pooling layers for spatial downsampling.
- Adaptive average pooling.
- Fully connected layers for classification.
- Dropout for regularization.
- Softmax to convert model outputs into class probabilities.

## 🛠️ Technologies Used

- Python
- PyTorch
- Torchvision
- Scikit-learn
- NumPy
- Matplotlib
- Hugging Face Datasets
- Gradio
- Pillow

## ⚙️ Project Workflow

1. Loaded and explored the PlantVillage dataset.
2. Split the dataset into training, validation, and testing sets.
3. Applied image preprocessing, resizing, normalization, and data augmentation.
4. Built a custom CNN architecture using PyTorch.
5. Trained the model and monitored training and validation performance.
6. Evaluated the model using classification metrics.
7. Saved the trained model checkpoint.
8. Developed an interactive Gradio application.
9. Deployed the application on Hugging Face Spaces.

## 📈 Model Performance

The model achieved the following results on the test set:

| Metric | Score |
|---|---:|
| Accuracy | 93.23% |
| Macro Precision | 91.48% |
| Macro Recall | 91.59% |
| Macro F1-Score | 91.25% |

These metrics provide an overview of the model's classification performance across the 38 classes.

## 💻 How to Use the Application

1. Open the [Live Demo](https://huggingface.co/spaces/EngySalem/plant-disease-classification).
2. Upload an image of a plant leaf.
3. Run the prediction.
4. Review the top 3 predicted classes and their confidence scores.

## 📂 Project Files

The deployment uses the following main files:

- `app.py` — Loads the trained model and provides the Gradio interface.
- `plant_disease_cnn.pth` — Saved trained model checkpoint.
- `requirements.txt` — Python dependencies required to run the application.

