#
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier("detectors/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("detectors/haarcascade_eye.xml")
cap = cv2.VideoCapture(0)

i = 0

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = eye_cascade.detectMultiScale(gray)
    eyeimages = []
    for (ex, ey, ew, eh) in eyes:
        eye = img[ey : ey + eh, ex : ex + ew]
        eyeimages.append(eye)
        cv2.imwrite("eyes/{}.png".format(i), eye)
        i += 1

    if eyeimages:
        cv2.imshow("img", eyeimages[0])
        k = cv2.waitKey(30) & 0xFF
        if k == 27:
            break

cap.release()
cv2.destroyAllWindows()
