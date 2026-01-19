🩺 Pneumonia Detection from Chest X-Ray using CNN (From Scratch)

A deep learning project that detects Pneumonia from chest X-ray images using a Convolutional Neural Network built entirely from scratch.
The repository includes the full training pipeline and an inference application for testing predictions on unseen X-ray images.

📌 Project Overview

Pneumonia is a serious lung infection that can be diagnosed using chest X-ray scans.
This project applies Computer Vision and Deep Learning to automatically classify X-ray images into:

Pneumonia

Normal

Instead of relying on transfer learning, the CNN architecture was manually designed and trained to better understand feature extraction in medical imaging.

🚀 Features

✔ Custom CNN model built from scratch
✔ Binary classification (Pneumonia vs Normal)
✔ Complete training pipeline
✔ Inference app for real-time predictions
✔ Modular and clean project structure
✔ Lightweight repository (dataset & trained weights excluded)

🧠 Tech Stack

Python

TensorFlow / Keras

NumPy

OpenCV

Matplotlib

Streamlit (for inference app)

🏗 Model Architecture

The CNN consists of multiple:

Convolution layers

ReLU activations

MaxPooling layers

Fully connected dense layers

Designed to capture spatial patterns specific to medical X-ray images.

📊 Training Pipeline

Image preprocessing & normalization

Dataset loading & labeling

CNN training from scratch

Validation & evaluation

Model saving for inference

🖥 Inference App

The project includes a simple app that allows users to:

Upload a chest X-ray image

Run prediction using the trained CNN

Display the result (Pneumonia / Normal)

📁 Dataset

This project uses a publicly available Chest X-Ray dataset (Pneumonia vs Normal).

⚠ Dataset is not included in the repo due to size.
Users must download it manually and place it inside the dataset folder.

📈 Future Improvements

Improve accuracy using deeper architecture

Add Grad-CAM visualization for interpretability

Deploy model on cloud

Convert to mobile-compatible version
