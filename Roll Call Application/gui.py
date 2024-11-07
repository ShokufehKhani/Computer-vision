import tkinter as tk
from tkinter import messagebox, simpledialog
import cv2
from PIL import Image, ImageTk
import os
import joblib
from face_recognition import detect_faces, preprocess_image, extract_hog_features
import numpy as np
from datetime import datetime

# Load pre-trained models
svm_model = joblib.load('svm_model.joblib')
lda = joblib.load('lda.joblib')
label_encoder = joblib.load('le.joblib')

class FaceRecognitionApp:
    def __init__(self, root, face_cascade, model, lda, label_encoder):
        self.root = root
        self.face_cascade = face_cascade
        self.model = model
        self.lda = lda
        self.label_encoder = label_encoder
        self.cap = None
        self.panel = None
        self.current_frame = None

        self.root.title("Employee Face Recognition System")
        self.root.geometry("800x600")

        self.create_main_menu()
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_main_menu(self):
        self.clear_frame()
        label = tk.Label(self.root, text="Welcome to the Employee Face Recognition System", font=("Arial", 16))
        label.pack(pady=20)

        btn_identify = tk.Button(self.root, text="Identify Employee for Entry", font=("Arial", 14), command=self.identify_employee)
        btn_identify.pack(pady=10)

        btn_add_user = tk.Button(self.root, text="Register a New Employee", font=("Arial", 14), command=self.add_employee)
        btn_add_user.pack(pady=10)

    def identify_employee(self):
        self.clear_frame()
        self.message_label = tk.Label(self.root, text="Please position yourself in front of the camera. Taking photo in 5 seconds...", font=("Arial", 14))
        self.message_label.pack(pady=20)
        self.panel = tk.Label(self.root)
        self.panel.pack()
        self.open_camera(capture_callback=self.capture_and_recognize)

    def add_employee(self):
        user_name = simpledialog.askstring("Input", "Enter the new employee's name:", parent=self.root)
        if user_name:
            self.clear_frame()
            user_path = os.path.join('Employee_Photos', user_name)
            os.makedirs(user_path, exist_ok=True)
            self.panel = tk.Label(self.root)
            self.panel.pack()
            self.message_label = tk.Label(self.root, text=f"Capturing images for {user_name}...", font=("Arial", 14))
            self.message_label.pack(pady=20)
            self.open_camera(capture_callback=lambda: self.capture_new_employee(user_name, user_path))

    def open_camera(self, capture_callback):
        self.release_camera()
        self.cap = cv2.VideoCapture(0)
        self.show_frame()
        self.root.after(5000, capture_callback)

    def release_camera(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def show_frame(self):
        if self.cap is not None:
            ret, frame = self.cap.read()
            if ret:
                self.current_frame = frame
                cv2image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                img = Image.fromarray(cv2image)
                imgtk = ImageTk.PhotoImage(image=img)
                self.panel.configure(image=imgtk)
                self.panel.image = imgtk
            self.root.after(10, self.show_frame)

    def capture_and_recognize(self):
        self.release_camera()
        self.message_label.config(text="Photo captured and processing...")

        if self.current_frame is not None:
            img_name = "captured_image.jpg"
            cv2.imwrite(img_name, self.current_frame)

            faces = detect_faces(self.current_frame, self.face_cascade)

            if len(faces) > 0:
                recognized = False

                for (x, y, w, h) in faces:
                    face = self.current_frame[y:y + h, x:x + w]
                    preprocessed_image = preprocess_image(face)
                    hog_features = extract_hog_features(preprocessed_image)
                    hog_features = hog_features.reshape(1, -1)
                    lda_features = self.lda.transform(hog_features)

                    prediction = self.model.predict(lda_features)
                    prediction_prob = self.model.predict_proba(lda_features)

                    most_similar_person = self.label_encoder.inverse_transform(prediction)[0]
                    max_prob = np.max(prediction_prob)

                    current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    if max_prob > 0.3:
                        result_text = f"Welcome, {most_similar_person}!\nDate and Time: {current_time}"
                        recognized = True
                        break

                if not recognized:
                    result_text = "Your face is not recognizable. Please ask admin to register your image."
                    self.message_label.config(text=result_text)
                    self.add_back_button_below_frame()
                    return

                self.show_result(preprocessed_image, result_text)
            else:
                self.message_label.config(text="No face detected. Please try again.")
                self.add_back_button_below_frame()
                return

        else:
            self.message_label.config(text="Failed to capture image. Please try again.")
            self.add_back_button()
            return

        self.add_back_button()

    def capture_new_employee(self, user_name, user_path):
        self.image_count = 1
        self.user_name = user_name
        self.user_path = user_path
        self.capture_next_image()

    def capture_next_image(self):
        if self.image_count <= 10:
            if self.image_count in [1, 2]:
                message = "Please look at the camera"
            elif self.image_count in [3, 4]:
                message = "Please look to the right"
            elif self.image_count in [5, 6]:
                message = "Please look to the left"
            elif self.image_count in [7, 8]:
                message = "Please look up"
            elif self.image_count in [9, 10]:
                message = "Please look down"

            self.message_label.config(text=message)
            self.root.update()

            ret, frame = self.cap.read()
            if ret:
                img_name = os.path.join(self.user_path, f"{self.image_count}.jpg")
                cv2.imwrite(img_name, frame)
                self.image_count += 1
                self.root.after(5000, self.capture_next_image)  
            else:
                self.message_label.config(text="Failed to capture image. Please try again.")
                self.add_back_button_below_frame()
        else:
            self.release_camera()
            self.message_label.config(text="Thank you for your time and patience. Your images have been captured successfully.")
            self.add_back_button_below_frame()

    def show_result(self, image, text):
        cv2image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(cv2image)
        imgtk = ImageTk.PhotoImage(image=img)
        self.panel.configure(image=imgtk)
        self.panel.image = imgtk
        self.message_label.config(text=text)

    def add_back_button(self):
        back_btn = tk.Button(self.root, text="Back to Main Menu", command=self.create_main_menu)
        back_btn.pack(pady=10)

    def add_back_button_below_frame(self):
        back_btn = tk.Button(self.root, text="Back to Main Menu", command=self.create_main_menu)
        back_btn.pack(side=tk.BOTTOM, pady=10)

    def clear_frame(self):
        self.release_camera()
        for widget in self.root.winfo_children():
            widget.destroy()

    def on_closing(self):
        if messagebox.askokcancel("Quit", "Do you want to quit?"):
            self.release_camera()
            self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    svm_model = joblib.load('svm_model.joblib')
    lda = joblib.load('lda.joblib')
    label_encoder = joblib.load('le.joblib')
    app = FaceRecognitionApp(root, face_cascade, svm_model, lda, label_encoder)
    root.mainloop()
