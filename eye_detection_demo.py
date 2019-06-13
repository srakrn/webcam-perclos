#
import numpy as np
import cv2

face_cascade = cv2.CascadeClassifier("detectors/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("detectors/haarcascade_eye.xml")
cap = cv2.VideoCapture(2)

while True:
    ret, img = cap.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    eyes = eye_cascade.detectMultiScale(gray)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), (0, 255, 0), 2)

    cv2.imshow("img", img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
