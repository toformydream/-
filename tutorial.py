from ultralytics import YOLO
from ultralytics.yolo.v8.pose.predict import DetectionPredictor
import cv2
# Load a model
#model = YOLO('yolov8n-pose.pt')  # load an official model
def main():
    model = YOLO('yolov8n-pose.pt')  # load a custom model
    img = cv2.imread('bus.jpg')
    # Predict with the model
    results = model(img, show=True)  # predict on an image
    cv2.waitKey()
    cv2.destroyAllWindows
    # Print joint locations
    for i, result in enumerate(results):
        print(f"Object {i}:")
        print(result.keypoints)

if __name__ == '__main__':
    main()