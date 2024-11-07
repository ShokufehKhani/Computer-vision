import os
import cv2
from skimage import exposure, feature
import numpy as np

IMAGE_SIZE = (400, 300)

def load_images(base_path, face_cascade):
    data = []
    labels = []

    # Iterate through each person's directory
    for person in os.listdir(base_path):
        person_path = os.path.join(base_path, person)
        if os.path.isdir(person_path):
            # Iterate through each image file in the person's directory
            for img_file in os.listdir(person_path):
                img_path = os.path.join(person_path, img_file)
                img = cv2.imread(img_path)
                if img is not None:
                    faces = detect_faces(img, face_cascade)
                    if len(faces) > 0:
                        (x, y, w, h) = faces[0]
                        face = img[y:y + h, x:x + w]
                        data.append(face)
                        labels.append(person)
    return data, labels

def preprocess_image(image):
    # Convert to grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Histogram equalization for better contrast
    equalized = exposure.equalize_hist(gray)
    # Resize to standard size
    resized = cv2.resize(equalized, IMAGE_SIZE)
    # Convert back to uint8 type
    return (resized * 255).astype(np.uint8)

def detect_faces(image, face_cascade):
    # Convert to grayscale for face detection
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    # Detect faces in the image
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    return faces

def extract_hog_features(image):
    # Extract Histogram of Oriented Gradients (HOG) features
    hog_features = feature.hog(image, pixels_per_cell=(8, 8), cells_per_block=(2, 2), visualize=False)
    return hog_features