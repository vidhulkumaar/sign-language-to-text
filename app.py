import cv2
import numpy as np
import json
import mediapipe as mp
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import tkinter as tk
from PIL import Image, ImageTk

# Load model and class indices
model = load_model("sign_model_fast.h5")
with open("class_indices.json") as f:
    class_indices = json.load(f)
labels = list(class_indices.keys())

# MediaPipe hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1)
mp_draw = mp.solutions.drawing_utils

# Tracking variables
ct = {label: 0 for label in labels}
ct['blank'] = 0
current_letter = ''
word = ''
sentence = ''
blank_flag = 0

# Initialize webcam
cap = cv2.VideoCapture(0)

# Setup GUI
root = tk.Tk()
root.title("Sign Language to Text")

video_label = tk.Label(root)
video_label.pack()

letter_label = tk.Label(root, text="Letter: ", font=("Helvetica", 20), fg="blue")
letter_label.pack()

word_label = tk.Label(root, text="Word: ", font=("Helvetica", 20), fg="green")
word_label.pack()

sentence_label = tk.Label(root, text="Sentence: ", font=("Helvetica", 20), fg="red")
sentence_label.pack()

def update_frame():
    global current_letter, word, sentence, blank_flag, ct

    ret, frame = cap.read()
    if not ret:
        return

    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_frame)

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            h, w, _ = frame.shape
            x_list = [int(lm.x * w) for lm in hand_landmarks.landmark]
            y_list = [int(lm.y * h) for lm in hand_landmarks.landmark]
            x_min, x_max = max(min(x_list) - 20, 0), min(max(x_list) + 20, w)
            y_min, y_max = max(min(y_list) - 20, 0), min(max(y_list) + 20, h)

            roi = frame[y_min:y_max, x_min:x_max]
            if roi.size == 0:
                continue

            try:
                roi_resized = cv2.resize(roi, (64, 64))
                roi_normalized = roi_resized.astype("float32") / 255.0
                roi_input = img_to_array(roi_normalized)
                roi_input = np.expand_dims(roi_input, axis=0)

                prediction = model.predict(roi_input)[0]
                index = np.argmax(prediction)
                current_letter = labels[index]
                ct[current_letter] += 1

                if ct[current_letter] > 40:
                    for label in labels:
                        if label != current_letter and abs(ct[current_letter] - ct[label]) <= 10:
                            break
                    else:
                        if current_letter.lower() == 'blank':
                            if not blank_flag and word:
                                sentence += word + " "
                                word = ''
                                blank_flag = 1
                        else:
                            blank_flag = 0
                            word += current_letter
                        ct = {label: 0 for label in labels}
                        ct['blank'] = 0

                # Update labels
                letter_label.config(text=f"Letter: {current_letter}")
                word_label.config(text=f"Word: {word}")
                sentence_label.config(text=f"Sentence: {sentence}")

            except Exception as e:
                print("Prediction Error:", e)

    # Convert image to Tkinter format and update GUI
    img = Image.fromarray(rgb_frame)
    imgtk = ImageTk.PhotoImage(image=img)
    video_label.imgtk = imgtk
    video_label.configure(image=imgtk)

    video_label.after(10, update_frame)

# Start webcam loop
update_frame()

# Close webcam on window close
def on_close():
    cap.release()
    root.destroy()

root.protocol("WM_DELETE_WINDOW", on_close)
root.mainloop()
