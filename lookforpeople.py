from ultralytics import YOLO
import cv2

model = YOLO('yolov8l.pt')



cap = cv2.VideoCapture(0) 

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

if not cap.isOpened():
    print("Error: Could not access the virtual camera.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to read frame from virtual camera.")
        break
    
    results = model(frame, imgsz=1080, conf=0.25, classes=[0])

    annotated_frame = results[0].plot() 

    cv2.imshow('YOLO Detection (1080p)', annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()