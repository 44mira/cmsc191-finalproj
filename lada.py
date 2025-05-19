import dlib
import cv2
import numpy as np

import streamlit as st

# constants  =================================================================

# for closing window
ESC = 27

# 3x3 kernel for convolutions
kernel = np.ones((3, 3), np.uint8)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# ============================================================================


def detect_face(img):
    # preprocess the image
    grayscaled = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(grayscaled, (5, 5), 0)
    opened = cv2.morphologyEx(blurred, cv2.MORPH_OPEN, kernel)
    processed = opened

    # detect faces
    faces = detector(processed)
    for face in faces:
        landmarks = predictor(processed, face)

        # higlight landmarks
        landmark_points = []
        for n in range(0, 68):
            x = landmarks.part(n).x
            y = landmarks.part(n).y
            landmark_points.append((x, y))

            cv2.circle(img, (x, y), 1, (0, 0, 255), -1)
        landmark_points = np.array(landmark_points, dtype=np.int32)
        cvx_hull = cv2.convexHull(landmark_points)

        cv2.polylines(img, [cvx_hull], True, (255, 0, 255), 2)

    return img


# sidebar configuration
with st.sidebar:
    st.write("# Settings")
    enabled_feed = st.toggle("Enable live feed")

    st.toggle(
        "Show Convex hull",
        key="enabled_cvxhull",
        disabled=enabled_feed,
    )
    st.toggle(
        "Show Landmarks",
        key="enabled_landmarks",
        disabled=enabled_feed,
    )
    st.toggle(
        "Show Mask",
        key="enabled_mask",
        disabled=enabled_feed,
    )
    st.toggle(
        "Show Delauney Triangles",
        key="enabled_tri",
        disabled=enabled_feed,
    )

# capture webcam
DISP = st.image([])
webcam = cv2.VideoCapture(0)

# read webcam
while enabled_feed:
    _, frame = webcam.read()

    frame = preprocess_face(frame)
    DISP.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

webcam.release()
