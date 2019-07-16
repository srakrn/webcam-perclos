import numpy as np
from numpy.linalg import norm
import cv2
import dlib
import argparse


def eye_ear_score(eye_landmarks):
    eye_landmarks = np.array(eye_landmarks)
    dh1 = eye_landmarks[1] - eye_landmarks[5]
    dh2 = eye_landmarks[2] - eye_landmarks[4]
    dv = eye_landmarks[0] - eye_landmarks[3]
    return (norm(dh1) + norm(dh2)) / (2 * norm(dv))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("filename", type=str)
    parser.add_argument("--videoout", type=str)
    parser.add_argument("--csvout", type=str)

    args = parser.parse_args()

    face_cascade = cv2.CascadeClassifier(
        "detectors/haarcascade_frontalface_default.xml"
    )
    eye_cascade = cv2.CascadeClassifier("detectors/haarcascade_eye.xml")
    detector = dlib.get_frontal_face_detector()
    predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")
    font = cv2.FONT_HERSHEY_SIMPLEX

    cap = cv2.VideoCapture(args.filename)

    frame_width = int(cap.get(3))
    frame_height = int(cap.get(4))
    if args.videoout:
        out = cv2.VideoWriter(
            args.videoout,
            cv2.VideoWriter_fourcc("M", "J", "P", "G"),
            30,
            (frame_width, frame_height),
        )

    if args.csvout:
        f = open(args.csvout, "w")

    while True:
        grabbed, img = cap.read()
        if not grabbed:
            break
        grayscale = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        faces = detector(grayscale)
        for face in faces[:1]:
            l, t, r, b = face.left(), face.top(), face.right(), face.bottom()
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
            average_ear = (left_ear + right_ear) / 2
            cond = "OPEN" if average_ear > 0.18 else "CLOSED"
            cond += " ({:.2f})".format(average_ear)
            cv2.putText(
                img, cond, (l, t + 20), font, 1, (255, 255, 255), 2, cv2.LINE_AA
            )
            if args.csvout:
                f.write("{},{}\n".format(cond, average_ear))
        cv2.imshow("Frame", img)
        if args.videoout:
            out.write(img)
        key = cv2.waitKey(1)
        if key == 27:
            break

    cap.release()
    if args.csvout:
        f.close()
    if args.videoout:
        out.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
