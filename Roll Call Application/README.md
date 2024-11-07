# **Employee Face Recognition Roll Call System**

This application is a face recognition-based roll call system for employees. The system detects if a person in front of the camera is an employee or not, and if recognized, logs their entry with their name, date, and time. The application is designed to work without deep learning models, making it lightweight and suitable for environments with limited resources.

## **Features**

- **Face Detection**: Detects faces in real-time using a webcam.
- **Employee Recognition**: Identifies employees based on pre-registered photos and verifies them against existing records.
- **Roll Call Logging**: Logs the entry of recognized employees with the date and time, displaying a welcome message.
- **Unregistered Users**: Alerts when an unrecognized face attempts entry.
- **Employee Registration**: Allows administrators to register new employees by capturing multiple images for reliable recognition.
- **Data Augmentation**: Generates augmented images for better accuracy, including flips, rotations, brightness adjustments, and translations.

## **File Structure**

- `data_augmentation.py`: Script for augmenting employee images to improve recognition accuracy.
- `face_detection.py`: Contains functions for face detection, preprocessing, and feature extraction.
- `model_training.py`: Trains a Support Vector Machine (SVM) model on employee facial features extracted via Histogram of Oriented Gradients (HOG).
- `gui.py`: Graphical User Interface built with Tkinter for user interaction and webcam-based recognition.
- `svm_model.joblib`, `lda.joblib`, `le.joblib`: Saved models and encoders for recognition (used in `gui.py`).

## **Setup Instructions**

### **Prerequisites**

1. **Python 3.x** with the following packages:
   - `opencv-python`
   - `numpy`
   - `scikit-image`
   - `scikit-learn`
   - `joblib`
   - `Pillow`
   - `tkinter`

2. **Haar Cascade for Face Detection**:
   - Download the pre-trained Haar Cascade XML file for face detection from [OpenCV GitHub](https://github.com/opencv/opencv/tree/master/data/haarcascades).
   - Place the XML file in the working directory.

### **Installation**

1. **Clone the Repository**:
   ```bash
   git clone https://github.com/yourusername/EmployeeFaceRecognition
   cd EmployeeFaceRecognition

2. **Prepare Dataset**:
- Create a folder named `Employee_Photos` with subfolders for each employee.
- Place each employee’s photos in their respective subfolder for training.

### **Model Training**
1. **Generate Augmented Images**:
- Run the `data_augmentation.py` script to create augmented images for each employee.

   ```bash
      python data_augmentation.py

2. **Train the SVM Model**:
- Run `model_training.py` to train the SVM model. The trained model, label encoder, and dimensionality reducer are saved as .joblib files.

   ```bash
      python model_training.py

### **Running the Application**

1. **Start the Application:**:
      ```bash
      python gui.py

2. **Using the Application**:

- **Identify Employee**: The employee stands in front of the camera, and the system will attempt to recognize and log their entry if recognized.
- **Register New Employee**: Admins can register new employees through the application, capturing images in various poses for accuracy.
Example Usage

### **Example Usage**

1. **Employee Identification**: Stand in front of the camera. If recognized, the system logs the entry.
2. **New Employee Registration**: Admins can use the interface to register new employees, capturing images for future identification.

### **Future Improvements**

Integrate with a remote database for centralized storage and management.
Implement additional face detection and recognition models for enhanced performance.
Optimize the GUI for better cross-platform compatibility.

### **Contributing**
Contributions are welcome! If you have ideas for new features, improvements, or bug fixes, feel free to open an issue or submit a pull request.

### **Important Note on Dataset**

The images used for training and testing the face recognition model are not included in this repository due to privacy concerns and the lack of permission from individuals to publicly share their images.

To use the system, you will need to provide your own dataset of employee photos. Please follow the instructions below to prepare the dataset:

1. **Prepare Dataset**:

- Create a folder named Employee_Photos with subfolders for each employee.
- Place each employee’s photos in their respective subfolder for training.
   ```bash
   Employee_Photos/
    Employee_1/
        1.jpg
        2.jpg
        ...
    Employee_2/
        1.jpg
        2.jpg
        ...
