import io
import torch
import cv2
import numpy as np
from PIL import Image
from fastapi import FastAPI, UploadFile, File
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pathlib import Path
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
import albumentations as A
from albumentations.pytorch import ToTensorV2
import base64

app = FastAPI(title="ASL Sign Language Detector")

# ── Constants ──────────────────────────────────────────────────────────────
CLASS_NAMES = [
    "__background__",
    "A", "B", "C", "D", "E", "F", "G", "H", "I", "J",
    "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T",
    "U", "V", "W", "X", "Y", "Z",
]
NUM_CLASSES = 27
IMG_SIZE = 640
CONFIDENCE_THRESHOLD = 0.5
MODEL_PATH = Path(__file__).resolve().parent.parent / "model" / "best_asl_fasterrcnn.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Model loading ──────────────────────────────────────────────────────────
def build_model(num_classes: int):
    model = fasterrcnn_resnet50_fpn(weights=None)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


model = build_model(NUM_CLASSES)
state = torch.load(str(MODEL_PATH), map_location=DEVICE, weights_only=True)
model.load_state_dict(state)
model.to(DEVICE)
model.eval()


# ── Transforms ─────────────────────────────────────────────────────────────
inference_transform = A.Compose([
    A.Resize(IMG_SIZE, IMG_SIZE),
    A.Normalize(mean=[0, 0, 0], std=[1, 1, 1]),
    ToTensorV2(),
])


# ── Helpers ────────────────────────────────────────────────────────────────
def run_inference(image_rgb: np.ndarray):
    """Run model inference on an RGB numpy image. Returns predictions list."""
    orig_h, orig_w = image_rgb.shape[:2]
    transformed = inference_transform(image=image_rgb)
    tensor = transformed["image"].float().unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        outputs = model(tensor)[0]

    boxes = outputs["boxes"].cpu().numpy()
    labels = outputs["labels"].cpu().numpy()
    scores = outputs["scores"].cpu().numpy()

    # Scale boxes back to original image dimensions
    sx = orig_w / IMG_SIZE
    sy = orig_h / IMG_SIZE

    predictions = []
    for box, label, score in zip(boxes, labels, scores):
        if score < CONFIDENCE_THRESHOLD:
            continue
        x1, y1, x2, y2 = box
        predictions.append({
            "label": CLASS_NAMES[int(label)],
            "score": round(float(score), 3),
            "box": [
                int(x1 * sx), int(y1 * sy),
                int(x2 * sx), int(y2 * sy),
            ],
        })
    return predictions


def draw_predictions(image_rgb: np.ndarray, predictions: list) -> np.ndarray:
    """Draw bounding boxes + labels on an RGB image copy."""
    img = image_rgb.copy()
    for pred in predictions:
        x1, y1, x2, y2 = pred["box"]
        label = pred["label"]
        score = pred["score"]
        color = (0, 255, 100)
        cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
        text = f'{label} {score:.2f}'
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(img, (x1, y1 - th - 8), (x1 + tw + 4, y1), color, -1)
        cv2.putText(img, text, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
    return img


def numpy_to_base64(image_rgb: np.ndarray) -> str:
    """Encode RGB numpy array to base64 JPEG string."""
    img_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)
    _, buf = cv2.imencode(".jpg", img_bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return base64.b64encode(buf.tobytes()).decode("utf-8")


# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def index():
    html_path = Path(__file__).resolve().parent / "static" / "index.html"
    return HTMLResponse(content=html_path.read_text(encoding="utf-8"))


@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image_np = np.array(image)

    predictions = run_inference(image_np)
    annotated = draw_predictions(image_np, predictions)
    annotated_b64 = numpy_to_base64(annotated)

    return JSONResponse({
        "predictions": predictions,
        "annotated_image": f"data:image/jpeg;base64,{annotated_b64}",
    })
