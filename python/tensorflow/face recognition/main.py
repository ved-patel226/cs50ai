import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import cv2

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)
font = cv2.FONT_HERSHEY_COMPLEX

model = load_model("model.h5")


def get_className(classNo):
    if classNo == 1:
        return "Ved"
    elif classNo == 0:
        return "Hiral"


while True:
    success, imgOriginal = cap.read()
    faces = facedetect.detectMultiScale(imgOriginal, 1.3, 5)
    for x, y, w, h in faces:

        crop_img = imgOriginal[y : y + h, x : x + w]  # crop
        img = cv2.resize(crop_img, (188, 188))  # resize
        img = img.reshape(1, 188, 188, 3)  # reshape
        img = img / 255.0  # normalize

        prediction = model.predict(img)
        classIndex = np.argmax(prediction)
        probabilityValue = np.max(prediction)

        # moms name
        if classIndex == 0:
            cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOriginal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(
                imgOriginal,
                str(get_className(classIndex)),
                (x, y - 10),
                font,
                0.75,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        # my name
        elif classIndex == 1:
            cv2.rectangle(imgOriginal, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.rectangle(imgOriginal, (x, y - 40), (x + w, y), (0, 255, 0), -2)
            cv2.putText(
                imgOriginal,
                str(get_className(classIndex)),
                (x, y - 10),
                font,
                0.75,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
        # confidence
        cv2.putText(
            imgOriginal,
            str(round(probabilityValue * 100, 2)) + "%",
            (180, 75),
            font,
            0.75,
            (255, 0, 0),
            2,
            cv2.LINE_AA,
        )
    cv2.imshow("Result", imgOriginal)
    k = cv2.waitKey(1)
    # press space to quit
    if k == ord(" "):
        break

cap.release()
cv2.destroyAllWindows()
