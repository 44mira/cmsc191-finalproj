# Facial Landmark Visualizer

## Features

This project shows a live feed with toggleable settings that visualize the
following:
- 68 facial landmarks as defined in `./shape_predictor_68_face_landmarks.dat` with `dlib`.
- The convex hull of the facial landmarks.
- The mask from the convex hull bitwise-AND'd with the webcam frame.
- Lastly, the *Delauney*  triangles in between the landmark points.

These steps can be observed in isolation or together with the other settings.
However, they must be set before the live feed is run.

## Running locally

This project requires either `uv` or some form of a Python virtual environment
manager.

### Running with `uv`

```bash
uv run streamlit run lada.py
```

### Running with `venv`

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

streamlit run lada.py
```
