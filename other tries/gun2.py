# from yolov5 import YOLOv5
# import cv2

# # Load YOLOv5 model (best.pt should be the trained model from YOLOv5)
# model = YOLOv5("best.pt")  # <-- Replace with your actual path

# # Open the webcam
# cap = cv2.VideoCapture(0)
# if not cap.isOpened():
#     print("Error: Could not open webcam.")
#     exit()

# while True:
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Run inference
#     results = model(frame)

#     # Visualize results
#     annotated_frame = results.render()[0]

#     # Show the frame with detections
#     cv2.imshow("Gun Detection - YOLOv5", annotated_frame)

#     # Press 'q' to exit
#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# cap.release()
# cv2.destroyAllWindows()
