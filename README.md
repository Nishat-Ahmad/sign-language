# ASL Sign Language Detector

This project is a FastAPI web app for detecting ASL alphabet signs from uploaded images or a live camera feed. The backend runs a Faster R-CNN model and returns bounding boxes plus predicted letter labels.

## Features

- Image upload and drag-and-drop detection
- Live camera mode in the browser
- FastAPI backend with a single-page HTML frontend
- Pretrained model weights included in the repository

## Project Structure

```text
signLanguage/
├── app/
│   ├── main.py           # FastAPI app and inference endpoint
│   └── static/
│       └── index.html    # Frontend UI served by the app
├── data/                 # Dataset or training data
├── model/
│   ├── best_asl_fasterrcnn.pth
│   └── final_asl_fasterrcnn.pth
├── scripts/
│   └── train_Faster_RCNN.ipynb
├── requirements.txt
└── README.md
```

## Requirements

Python 3.10+ is recommended.

Install the dependencies with:

```bash
pip install -r requirements.txt
```

## Run the Web App

From the project root, start the server with:

```bash
uvicorn app.main:app --reload
```

If `uvicorn` is not on your PATH, use:

```bash
python -m uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

## How It Works

The app loads the model weights from `model/best_asl_fasterrcnn.pth` on startup. It accepts an uploaded image at `/predict`, runs inference, and returns:

- predicted ASL letter labels
- confidence scores
- an annotated image with bounding boxes

The browser UI lets you:

- upload an image for detection
- use the camera mode for live capture
- view the annotated result and detected letters

## Model Files

The repository already includes trained weights in `model/`.

- `best_asl_fasterrcnn.pth` is the default file loaded by the app
- `final_asl_fasterrcnn.pth` is an additional saved checkpoint

If you retrain the model, keep the updated weights in the same folder or update `MODEL_PATH` in `app/main.py`.

## Notes

- The project expects the weights file to exist before launch.
- The app is designed for ASL alphabet detection, not full sign language translation.
- `app/static/index.html` contains the complete frontend served by FastAPI.

## Training

Training-related work appears to live in `scripts/train_Faster_RCNN.ipynb`. If you want to retrain the model, start there and save the resulting weights into `model/`.
