import numpy as np
import cv2
import eyelib
import math

eye_cascade = cv2.CascadeClassifier("detectors/haarcascade_eye_tree_eyeglasses.xml")
# cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("eyes/videos/1.mp4")
i = 1
while True:
    ret, img = cap.read()
    print(img.shape)
    eyes = eye_cascade.detectMultiScale(img)
    for (ex, ey, ew, eh) in eyes:
        ex, ey, ew, eh = eyes[0]
        cv2.rectangle(img, (ex, ey), (ex + ew, ey + eh), 3)
        roi = img[ey : ey + eh, ex : ex + ew]
        upper_eye, lower_eye = eyelib.find_eye_position(roi)
        if type(upper_eye) != type(None):
            upper_eye = eyelib.move_second_degree(upper_eye, ex, ey + int(eh * 0.2))
            lower_eye = eyelib.move_second_degree(lower_eye, ex, ey + int(eh * 0.2))
            upper_eye = np.poly1d(upper_eye)
            lower_eye = np.poly1d(lower_eye)
            y_l, y_r = eyelib.solve_second_degree(upper_eye, lower_eye)
            if not (math.isnan(y_l) or math.isnan(y_r)):
                x_l, x_r = upper_eye(y_l), upper_eye(y_r)
                y_l, y_r, x_l, x_r = int(y_l), int(y_r), int(x_l), int(x_r)
                cv2.circle(img, (y_l, x_l), 3, (0, 0, 255), -1)
                cv2.circle(img, (y_r, x_r), 3, (0, 255, 0), -1)
    # for ends here
    cv2.imshow("img", img)
    cv2.imwrite("eyes/{}.png".format(i), img)
    i += 1
    k = cv2.waitKey(30) & 0xFF
    if k == 27:
        break

cap.release()
cv2.destroyAllWindows()
