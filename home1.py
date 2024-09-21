import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox
from PIL import Image, ImageTk
from keras.models import load_model

# Load the pre-trained drowsiness detection model
model = load_model(r'C:\Users\bhara\OneDrive\Desktop\eye_detection\drowsiness_new6.h5')

# Haar Cascade for face detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


image_path = r'C:\Users\bhara\OneDrive\Desktop\eye_detection\img1.jpg'
video_path = r'C:\Users\bhara\OneDrive\Desktop\eye_detection\video123.mp4'


print(f"Expected input shape: {model.input_shape}")


def process_frame(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    sleeping_people = 0
    age_predictions = []

    for (x, y, w, h) in faces:
       
        face = frame[y:y + h, x:x + w]

        
        face_resized = cv2.resize(face, (224, 224))  
        face_rgb = cv2.cvtColor(face_resized, cv2.COLOR_BGR2RGB)
        input_frame = np.expand_dims(face_rgb / 255.0, axis=0)  

        print(f"Processed input shape: {input_frame.shape}")  

        
        try:
            predictions = model.predict(input_frame)
        except ValueError as e:
            print(f"Error during model prediction: {e}")
            continue  

        drowsiness_score, age_prediction = predictions[0][0], predictions[0][1]

        
        if drowsiness_score > 0.5:  
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)  # Red for drowsy
            sleeping_people += 1
            age_predictions.append(int(age_prediction))
        else:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)  # Green for awake

       
        cv2.putText(frame, f"Age: {int(age_prediction)}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2)

    return frame, sleeping_people, age_predictions



def load_image():
    img = cv2.imread(image_path)
    img, sleeping_people, age_predictions = process_frame(img)

    
    display_image(img)

   
    if sleeping_people > 0:
        messagebox.showinfo("Detection Result", f"Sleeping people: {sleeping_people}\nAges: {', '.join(map(str, age_predictions))}")
    else:
        messagebox.showinfo("Detection Result", "No one is sleeping.")


def load_video():
    cap = cv2.VideoCapture(video_path)
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame, sleeping_people, age_predictions = process_frame(frame)

     
        display_image(frame)

    
    if sleeping_people > 0:
        messagebox.showinfo("Detection Result", f"Sleeping people: {sleeping_people}\nAges: {', '.join(map(str, age_predictions))}")
    else:
        messagebox.showinfo("Detection Result", "No one is sleeping.")

    cap.release()


def display_image(img):
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    img_tk = ImageTk.PhotoImage(img_pil)

    
    panel.config(image=img_tk)
    panel.image = img_tk


root = tk.Tk()
root.title("Drowsiness and Age Detection")

panel = tk.Label(root)  
panel.pack()


btn_image = tk.Button(root, text="Process Predefined Image", command=load_image)
btn_image.pack()

btn_video = tk.Button(root, text="Process Predefined Video", command=load_video)
btn_video.pack()


root.mainloop()
