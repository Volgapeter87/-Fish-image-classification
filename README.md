# Multiclass Fish Image Classification 🐟

This project focuses on classifying fish images into multiple categories using Deep Learning models. The system trains a Convolutional Neural Network (CNN) from scratch and compares it with several Transfer Learning models such as VGG16, ResNet50, MobileNet, InceptionV3, and EfficientNetB0. The goal is to determine the best performing model for fish species classification and deploy the final model through a Streamlit web application that allows users to upload an image and receive real-time predictions.

---

## Project Overview

Fish species classification is useful in marine research, seafood industries, automated monitoring systems, and educational applications. In this project, a dataset of fish images is used to train deep learning models capable of identifying different species from images. The workflow includes data preprocessing, model training, evaluation, comparison, and deployment.

---

## Technologies Used

- Python  
- TensorFlow / Keras  
- NumPy  
- Matplotlib  
- Seaborn  
- Scikit-learn  
- Streamlit  
- Jupyter Notebook  

---

## Dataset

The dataset contains images of different fish species organized into folders. Each folder represents one class label.

Example structure:

```
Dataset/
│
├── train/
├── val/
└── test/
```

Fish classes used in the dataset:

- animal fish  
- animal fish bass  
- fish sea_food black_sea_sprat  
- fish sea_food gilt_head_bream  
- fish sea_food hourse_mackerel  
- fish sea_food red_mullet  
- fish sea_food red_sea_bream  
- fish sea_food sea_bass  
- fish sea_food shrimp  
- fish sea_food striped_red_mullet  
- fish sea_food trout  

---

## Project Workflow

```
Dataset Collection
        ↓
Data Preprocessing
        ↓
Image Augmentation
        ↓
CNN Model Training
        ↓
Transfer Learning Models
        ↓
Model Evaluation
        ↓
Model Comparison
        ↓
Best Model Selection
        ↓
Streamlit Deployment
```

---

## Data Preprocessing

The following preprocessing techniques were applied to improve model performance:

- Image resizing to **224 × 224**
- Pixel normalization (**0–1 scaling**)
- Data augmentation:
  - Rotation
  - Zoom
  - Horizontal flip

These techniques help reduce overfitting and improve generalization.

---

## Models Implemented

This project compares six deep learning models:

1. CNN (Custom Model)  
2. VGG16  
3. ResNet50  
4. MobileNet  
5. InceptionV3  
6. EfficientNetB0  

Transfer learning models were initialized with **ImageNet weights** and fine-tuned on the dataset.

---

## Model Evaluation

Models were evaluated using the following metrics:

- Accuracy  
- Precision  
- Recall  
- F1 Score  
- Confusion Matrix  

Visualization tools used:

- Accuracy comparison charts  
- Confusion matrix heatmap  
- Classification report  

---

## Model Performance

After comparing all models, the **CNN model achieved the best performance**.

Example result:

```
CNN Accuracy ≈ 97%
```

The CNN model was selected as the **final model for deployment**.

---

## Streamlit Web Application

A Streamlit web application was developed to perform real-time fish classification.

Features of the app:

- Upload fish image  
- Predict fish species  
- Display confidence score  
- Detect unknown species  

Run the application using:

```
streamlit run app.py
```

---

## Project Structure

```
Fish_image_classification
│
├── models
│   └── cnn_model.keras
│
├── cnn_training.py
├── transfer_learning_training.py
├── evaluate_models.py
├── model_comparison.ipynb
├── app.py
│
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Installation

Clone the repository:

```
git clone https://github.com/Volgapeter87/-Fish-image-classification.git
```

Navigate to the project folder:

```
cd -Fish-image-classification
```

Install dependencies:

```
pip install -r requirements.txt
```

---

## Running the Project

Run the Streamlit app:

```
streamlit run app.py
```

Upload an image of a fish and the model will predict the species.

---

## Example Prediction

```
Predicted Fish Species: Sea Bass
Confidence Score: 0.92
```

---

## Future Improvements

Possible improvements include:

- Adding more fish species to the dataset
- Improving dataset diversity
- Training on larger datasets
- Deploying the model on cloud platforms
- Building a mobile application for fish identification

---

