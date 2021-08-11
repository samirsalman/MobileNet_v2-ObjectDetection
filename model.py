import cv2
import numpy as np


class MobileNet:
    def __init__(self, pbtext_path: str, weights_path: str, classnames_path: str, width: int = 320, height: int = 320,
                 scale_ratio: float = 127.5, mean: float = 127.5):
        # mobileNet class names
        self.classNames = []
        classFile = classnames_path

        # read class names from file
        with open(classFile, "rt") as f:
            self.classNames = f.read().rstrip("\n").split("\n")

        # init a cv2 detection model from files
        self.model = cv2.dnn_DetectionModel(
            weights_path,
            pbtext_path)

        # model parameters
        self.model.setInputSize(width, height)
        self.model.setInputScale(1.0 / scale_ratio)
        self.model.setInputMean((mean, mean, mean))
        self.model.setInputSwapRB(True)

    def detect(self, img: np.ndarray) -> np.ndarray:

        # detect objects in a frame using the model (output label, confidence and box coordinates)
        classIds, confidences, bbox = self.model.detect(img, confThreshold=0.5)
        # if there is at least 1 object in the frame
        if len(classIds) != 0:
            for classId, confidence, box in zip(classIds.flatten(), confidences.flatten(), bbox):
                # draw rectangle
                cv2.rectangle(img, box, color=(0, 255, 0), thickness=3)
                # write class name
                cv2.putText(img, self.classNames[classId - 1], (box[0] + 10, box[1] + 30), cv2.FONT_HERSHEY_COMPLEX, 1,
                            (0, 255, 0),
                            2)

        return img

    def live_test(self, video: str = None, camera: bool = False):

        # open web cam stream
        if video is None or camera:
            cap = cv2.VideoCapture(0)
        else:
            # Open the video with opencv
            cap = cv2.VideoCapture(video)

        # Play the video
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = self.detect(img=frame)
            cv2.imshow('LiveTest', frame)

            key = cv2.waitKey(1)
            if key == 27:
                break

        cap.release()
        cv2.destroyAllWindows()
