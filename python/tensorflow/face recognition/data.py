import cv2
import os
from termcolor import cprint

video = cv2.VideoCapture(0)

facedetect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

count = 0

nameID = str(input("Enter Your Name: ")).lower()

path = "images/" + nameID


while True:
    if os.path.exists(path):
        cprint("Name Already Taken", "red", attrs=["bold"])
        nameID = str(input("Enter Your Name Again: "))
    else:
        cprint("Cool! Booting up now...", "blue", attrs=["bold"])
        os.makedirs(path)
        break

while True:
    ret, frame = video.read()
    faces = facedetect.detectMultiScale(frame, 1.3, 5)
    for x, y, w, h in faces:
        count = count + 1
        name = "./images/" + nameID + "/" + str(count) + ".jpg"
        cprint("Creating Images........." + name, "green", attrs=["bold"])
        cv2.imwrite(name, frame[y : y + h, x : x + w])
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 3)
    cv2.imshow("WindowFrame", frame)
    cv2.waitKey(1)
    if count >= 500:
        break
video.release()
cv2.destroyAllWindows()
