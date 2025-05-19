import dlib
import cv2
import numpy as np

import streamlit as st

# constants  =================================================================

# for closing window
ESC = 27

# 3x3 kernel for convolutions
kernel = np.ones((3, 3), np.uint8)

# use dlib for the haar cascades
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor("shape_predictor_68_face_landmarks.dat")

# ============================================================================


def get_cvx_hull(landmark_points):
    # obtaining the convex hull
    landmark_points = np.array(landmark_points, dtype=np.int32)
    cvx_hull = cv2.convexHull(landmark_points)

    return cvx_hull, landmark_points


def get_cvx_mask(cvx_hull, processed_img):
    # obtaining the mask from the convex hull
    zs = np.zeros_like(processed_img)
    mask = cv2.fillConvexPoly(zs, cvx_hull, (255, 255, 255))

    return mask


def get_triangles(cvx_hull, points):
    # obtaining the delauney triangles within a convex hull
    bounding_rect = cv2.boundingRect(cvx_hull)
    subdivisions = cv2.Subdiv2D(bounding_rect)
    subdivisions.insert(points)

    triangles = subdivisions.getTriangleList()

    return np.array(triangles, dtype=np.int32)


def preprocess_face(img):
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

            if st.session_state.enabled_landmarks:
                cv2.circle(img, (x, y), 2, (0, 0, 255), -1)

        cvx_hull, points = get_cvx_hull(landmark_points)
        mask = get_cvx_mask(cvx_hull, processed)
        triangles = get_triangles(cvx_hull, landmark_points)

        # toggleable displays =================================================
        if st.session_state.enabled_cvxhull:
            cv2.polylines(img, [cvx_hull], True, (255, 0, 255), 2)

        if st.session_state.enabled_mask:
            img = cv2.bitwise_and(img, img, mask=mask)

        if st.session_state.enabled_tri:
            # draw each triangle
            for t in triangles:
                settings = [(155, 0, 25), 1]
                pt1 = (t[0], t[1])
                pt2 = (t[2], t[3])
                pt3 = (t[4], t[5])

                cv2.line(img, pt1, pt2, *settings)
                cv2.line(img, pt2, pt3, *settings)
                cv2.line(img, pt3, pt1, *settings)

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
