import cv2
from ultralytics import YOLO


def main():
    model = YOLO('yolov8n-pose.pt')

    video_path = 0
    cap = cv2.VideoCapture(video_path)
    print("1")

    while cap.isOpened():
        print("2")
        success, frame = cap.read()
        if success:
            print("success")
            results = model(frame, save=True)

            annotated_frame = results[0].plot()
            cv2.imshow("YOLOv8 Inference", annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
              break
        else:
            print("fale")
            break
    cap.release()
    cv2.destroyAllWindows


if __name__ == '__main__':
    main()
