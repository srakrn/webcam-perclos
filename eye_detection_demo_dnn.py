import numpy as np
from numpy.linalg import norm
import cv2
import dlib

detector = dlib.cnn_face_detection_model_v1("mmod_human_face_detector.dat")
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture("eyes/videos/WIN_20190701_16_03_32_Pro.mp4")
font = cv2.FONT_HERSHEY_SIMPLEX


def eye_ear_score(eye_landmarks):
    eye_landmarks = np.array(eye_landmarks)
    dh1 = eye_landmarks[1] - eye_landmarks[5]
    dh2 = eye_landmarks[2] - eye_landmarks[4]
    dv = eye_landmarks[0] - eye_landmarks[3]
    return (norm(dh1) + norm(dh2)) / (2 * norm(dv))


f = open("logs.csv", "w")

while True:
    grabbed, img = cap.read()
    if not grabbed:
        break
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(grayscale)
    for face in faces[:1]:
        l, t, r, b = face.rect.left(), face.rect.top(), face.rect.right(), face.rect.bottom()
        cv2.rectangle(img, (l, t), (r, b), (255, 0, 0), 5)
        landmarks = predictor(grayscale, face)
        left_eyes = []
        right_eyes = []
        for i in range(36, 42):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            left_eyes.append([x, y])
        for i in range(42, 48):
            x, y = landmarks.part(i).x, landmarks.part(i).y
            right_eyes.append([x, y])
        for (x, y) in left_eyes:
            cv2.circle(img, (x, y), 2, (0, 255, 0), -1)
        for (x, y) in right_eyes:
            cv2.circle(img, (x, y), 2, (0, 0, 255), -1)
        left_ear, right_ear = eye_ear_score(left_eyes), eye_ear_score(right_eyes)
        average_ear = (left_ear + right_ear)/2
        cond = "OPEN" if average_ear > 0.18 else "CLOSED"
        cond += " ({:.2f})".format(average_ear)
        cv2.putText(img, cond, (l, t + 20), font, 1, (255, 255, 255), 2, cv2.LINE_AA)
        f.write("{},{}\n".format(cond, average_ear))
    cv2.imshow("Frame", img)
    key = cv2.waitKey(1)
    if key == 27:
        f.close()
        break

cap.release()
cv2.destroyAllWindows()
