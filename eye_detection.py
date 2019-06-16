import numpy as np
import cv2
import eyelib

eye_cascade = cv2.CascadeClassifier("detectors/haarcascade_eye.xml")
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("eyes/videos/1.mp4")

while True:
    ret, img = cap.read()

    eyes = eye_cascade.detectMultiScale(img)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), 3)
        roi = img[ey : ey + eh, ex : ex + ew]
        upper_eye, lower_eye = eyelib.find_eye_position(roi)
        if type(upper_eye) != type(None):
            upper_eye = np.poly1d(upper_eye)
            lower_eye = np.poly1d(lower_eye)
            y_l, y_r = eyelib.solve_second_degree(upper_eye, lower_eye)
            x_l, x_r = upper_eye(y_l), upper_eye(y_r)
            y_l, y_r, x_l, x_r = int(y_l), int(y_r), int(x_l), int(x_r)
            cv2.circle(img, (x_l, y_l), 3, (0, 0, 255), -1)
            cv2.circle(img, (x_r, y_r), 3, (0, 255, 0), -1)
    cv2.imshow("img", img)
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
