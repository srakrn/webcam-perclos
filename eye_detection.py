import numpy as np
import cv2

eye_cascade = cv2.CascadeClassifier("detectors/haarcascade_eye.xml")
cap = cv2.VideoCapture(0)

while True:
    ret, img = cap.read()

    eyes = eye_cascade.detectMultiScale(img)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), 3)
    cv2.imshow("img", img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
