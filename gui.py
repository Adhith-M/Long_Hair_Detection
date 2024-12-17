import tkinter as tk
from tkinter import filedialog, messagebox
import cv2
import numpy as np
import tensorflow as tf

# Load the pre-trained models
age_gender_model = tf.keras.models.load_model('Age_Sex_Detection.keras')  # Update with your model path
hair_length_model = tf.keras.models.load_model('hair_length_detection_model.keras')  # Update with your model path

def detect_hair_length(image):
    if image is None or image.size == 0:
        raise ValueError("Invalid image. The image is empty or could not be loaded.")
    img_resized = cv2.resize(image, (48, 48))  
    img_array = np.expand_dims(img_resized / 255.0, axis=0)
    prediction = hair_length_model.predict(img_array)
    return 'long' if prediction[0][0] > 0.5 else 'short'

def predict_age_gender(image):
    if image is None or image.size == 0:
        raise ValueError("Invalid image. The image is empty or could not be loaded.")
    
    img_resized = cv2.resize(image, (48, 48))  
    img_array = np.expand_dims(img_resized / 255.0, axis=0)  
    
    # Get the model prediction
    prediction = age_gender_model.predict(img_array)
    
    # Debugging output
    print(f"Model prediction output: {prediction}")

    if len(prediction) == 2:
        # Extracting and rounding the scalar age value
        age = int(np.round(prediction[1][0][0])) 

        # Extracting and rounding the scalar gender value
        gender_value = int(np.round(prediction[0][0][0]))  
        
        sex_f = ['Male', 'Female']
        
        gender = sex_f[gender_value]
        
    else:
        raise ValueError(f"Unexpected model output shape: {prediction.shape}")
    
    return age, gender

def classify_person(image):
    if image is None or image.size == 0:
        raise ValueError("Invalid image. The image is empty or could not be loaded.")
    
    age, gender = predict_age_gender(image)

    if 20 <= age <= 30:
        hair_length = detect_hair_length(image)
        if hair_length == 'long':
            return 'female'
        else:
            return 'male'
    
    return gender

def load_image():
    file_path = filedialog.askopenfilename()
    
    if file_path:
        image = cv2.imread(file_path)
        try:
            result = classify_person(image)
            result_label.config(text=f"The detected person is classified as: {result}")
        except ValueError as e:
            messagebox.showerror("Error", str(e))

# GUI Setup
root = tk.Tk()
root.title("Gender & Age Detection and Hair Length Classification")
root.geometry("700x500")  # Set window size to 700x500

label = tk.Label(root, text="Select an Image:")
label.pack(pady=20)

select_button = tk.Button(root, text="Browse", command=load_image)
select_button.pack(pady=10)

result_label = tk.Label(root, text="")
result_label.pack(pady=20)

root.mainloop()
