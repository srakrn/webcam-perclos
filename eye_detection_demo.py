import numpy as np
import cv2
import dlib

face_cascade = cv2.CascadeClassifier("detectors/haarcascade_frontalface_default.xml")
eye_cascade = cv2.CascadeClassifier("detectors/haarcascade_eye.xml")
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
cap = cv2.VideoCapture(0)
cap = cv2.VideoCapture("eyes/videos/1.mp4")

i = 0

while True:
    ret, img = cap.read()
    grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    faces = detector(grayscale)
    for face in faces:
        l, t, r, b = face.left(), face.top(), face.right(), face.buttom()
        cv2.rectangle(img, (l, t), (r, b), (255, 0 , 0), 5)
        landmarks = predictor(grayscale, face)
        for i in range(68):
            x, y = landmarks.part(n).x, landmarks.part(n).y
            cv2.circle(frame, (x, y), 4, (0, 255, 0), -1)
    
    cv2.imshow("Frame", img)
    key = cv2.waitKey(1)
    if key == 27:
        break

cap.release()
cv2.destroyAllWindows()
