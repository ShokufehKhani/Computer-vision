Rock-Paper-Scissors YOLOv8: Fine-Tuning and Real-Time Object Detection
* Overview
This repository contains the complete workflow for fine-tuning the YOLOv8 model on a Rock-Paper-Scissors dataset and using the trained model for real-time object detection. The project is structured into two main parts:

* Fine-Tuning the YOLOv8 Model
Using the Fine-Tuned Model for Real-Time and Image-Based Detection

* Project Workflow
1. Fine-Tuning YOLOv8 on Rock-Paper-Scissors Dataset
Description
In this first step, the YOLOv8s model was fine-tuned on a custom Rock-Paper-Scissors dataset to accurately detect and classify hand gestures. The process involved:

Dataset Preparation: Preparing the Rock-Paper-Scissors images with appropriate annotations.
Model Training: Fine-tuning the YOLOv8s model on the dataset.
Evaluation: Validating the model to ensure high accuracy and reliability.

2. Real-Time and Image-Based Object Detection
Description
After fine-tuning the model, the best-performing weights were downloaded and utilized for real-time detection using a webcam, as well as for detecting objects in static images. This step demonstrates the practical application of the fine-tuned YOLOv8 model.

Real-Time Detection: Implementing the trained YOLOv8 model for detecting Rock-Paper-Scissors gestures via a webcam feed.
Image Detection: Using the model to detect and classify objects in provided static images.
