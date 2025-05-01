from ultralytics import YOLO
import cv2
import time
import winsound  # Built-in for Windows to generate beep sound

# Load the trained model
model = YOLO("best_gun_yolo8.pt")  # <-- Replace with your actual path

# Open the webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

gun_detected = False
detection_start_time = None

# Set the window title and some UI elements
cv2.namedWindow("Gun Detection - YOLOv8", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Gun Detection - YOLOv8", 800, 600)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model.predict(source=frame, save=False, imgsz=416, conf=0.25)

    # Visualize results (bounding boxes and labels handled by the model itself)
    annotated_frame = results[0].plot()

    # Process detections
    if len(results[0].boxes) > 0:
        for box in results[0].boxes:
            label = box.cls[0]
            confidence = box.conf[0]

            if label == 0 and confidence > 0.25:  # Gun class, usually 0 or according to your model class mapping
                if not gun_detected:
                    # Mark the start of gun detection
                    gun_detected = True
                    detection_start_time = time.time()

                # Check if gun has been detected for more than 1 second
                if detection_start_time is not None and (time.time() - detection_start_time) >= 1:
                    # Play a beep sound (Windows built-in)
                    winsound.Beep(1000, 500)  # Frequency: 1000 Hz, Duration: 500 ms
                    detection_start_time = None  # Reset detection timer
            else:
                # If no gun is detected, reset flag
                gun_detected = False

    # Show the frame with detections
    cv2.imshow("Gun Detection - YOLOv8", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()
