import cv2
import time
import winsound  # Built-in for Windows to generate beep sound
import tkinter as tk
from PIL import Image, ImageTk
from ultralytics import YOLO
import pygame  # For playing music

# Load the trained YOLO model
model = YOLO("best_gun_yolo8.pt")  # <-- Replace with your actual path

# Open the webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

gun_detected = False
detection_start_time = None

# Initialize the Tkinter window
root = tk.Tk()
root.title("Gun Detection - YOLOv8")
root.geometry("800x650")  # Increased height for status and labels

# Create a canvas to display the webcam feed
canvas = tk.Canvas(root, width=800, height=600)
canvas.pack()

# Create a label to display detection status
status_label = tk.Label(root, text="No gun detected", font=("Helvetica", 16), bg="black", fg="white")
status_label.pack(fill=tk.X)

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Function to display the video feed
def update_frame():
    global gun_detected, detection_start_time

    # Read a frame from the webcam
    ret, frame = cap.read()
    if not ret:
        return

    # Run inference on the frame
    results = model.predict(source=frame, save=False, imgsz=416, conf=0.25)

    # Visualize results (bounding boxes and labels handled by the model itself)
    annotated_frame = results[0].plot()

    # Process detections
    gun_detected_local = False
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            label = box.cls[0]
            confidence = box.conf[0]

            if label == 0 and confidence > 0.50:  # Gun class with confidence > 50%
                gun_detected_local = True
                if not gun_detected:
                    # Mark the start of gun detection
                    gun_detected = True
                    detection_start_time = time.time()

                # Check if gun has been detected for more than 1 second
                if detection_start_time is not None and (time.time() - detection_start_time) >= 0.5:
                    # Play a beep sound (Windows built-in)
                    winsound.Beep(1000, 500)  # Frequency: 1000 Hz, Duration: 500 ms
                    # Play music
                    pygame.mixer.music.load("detection_alert.mp3")  # Replace with your audio file
                    pygame.mixer.music.play()  # Play the music
                    detection_start_time = None  # Reset detection timer
            else:
                # If no gun is detected, reset flag
                gun_detected_local = False

    # Update status label
    if gun_detected_local:
        status_label.config(text="Gun Detected!", bg="red", fg="white")
    else:
        status_label.config(text="No gun detected", bg="black", fg="white")
        gun_detected = False  # Reset flag if no gun is detected

    # Convert the OpenCV frame (BGR) to PIL (RGB)
    frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
    img = Image.fromarray(frame_rgb)
    img_tk = ImageTk.PhotoImage(image=img)

    # Update the canvas with the new image
    canvas.create_image(0, 0, anchor="nw", image=img_tk)
    canvas.image = img_tk  # Keep a reference to avoid garbage collection

    # Repeat this function after 20 ms to create the video loop
    root.after(20, update_frame)

# Start the video feed and update the Tkinter window
update_frame()

# Start the Tkinter main loop
root.mainloop()

# Release the webcam when the window is closed
cap.release()
cv2.destroyAllWindows()
