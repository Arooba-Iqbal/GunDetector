from ultralytics import YOLO
import cv2

# Load the trained model
model = YOLO("best_gun_yolo8.pt")  # <-- Replace with your actual path

# Open the webcam (0 = default camera)
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Run inference on the frame
    results = model.predict(source=frame, save=False, imgsz=416, conf=0.25)

    # Visualize results
    annotated_frame = results[0].plot()

    # Show the frame with detections
    cv2.imshow("Gun Detection - YOLOv8", annotated_frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release webcam and close window
cap.release()
cv2.destroyAllWindows()
