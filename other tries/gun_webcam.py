import torch
import cv2

# Load your trained model
model = torch.hub.load('ultralytics/yolov5', 'custom', path='best_cleaned.pt', force_reload=True)
#model = torch.hub.load('ultralytics/yolov5', 'custom', path='best.pt')  # Make sure best.pt is in same folder
model.conf = 0.4  # Set confidence threshold

# Start webcam
cap = cv2.VideoCapture(0)  # 0 is default webcam

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run detection
    results = model(frame)

    # Render results on the frame
    annotated_frame = results.render()[0]

    # Show the frame
    cv2.imshow('Gun Detector - Press Q to quit', annotated_frame)

    # Exit on Q key
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
