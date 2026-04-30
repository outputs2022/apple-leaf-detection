from fastapi import FastAPI, File, UploadFile, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse # Added for proper image serving
import numpy as np
import cv2
import io
import os
import uuid
import time
import threading
from PIL import Image
from tensorflow.keras.models import load_model

app = FastAPI()

# --- 1. BYPASS MIDDLEWARE & CORS ---
# This forces every response to include the headers that stop tunnels from blocking requests
@app.middleware("http")
async def add_tunnel_bypass_headers(request: Request, call_next):
    response = await call_next(request)
    response.headers["bypass-tunnel-reminder"] = "true" # For localtunnel
    response.headers["ngrok-skip-browser-warning"] = "true" # For ngrok
    response.headers["Access-Control-Allow-Origin"] = "*"
    return response

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

OUTPUT_DIR = "annotated_outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- 2. CUSTOM IMAGE ROUTE (Replaces StaticFiles) ---
# We use this instead of app.mount to ensure the image/jpeg header is forced
@app.get("/outputs/{filename}")
async def get_image(filename: str):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.exists(file_path):
        return FileResponse(file_path, media_type="image/jpeg")
    return Response(status_code=404)

# --- 3. MODEL CONFIGURATION ---
CLASS_LABELS   = ["Apple_healthy", "Apple_scab", "Black_rot", "Cedar_apple_rust"]
DISPLAY_LABELS = ["Healthy", "Apple Scab", "Black Rot", "Cedar Rust"]
SCORE_KEYS     = ["Apple_healthy", "Apple_scab", "Black_rot", "Cedar_apple_rust"]

IMG_SIZE  = 256
HSV_LOWER = np.array([20, 40, 40])
HSV_UPPER = np.array([90, 255, 255])

print("Loading model...")
model = load_model("modelAppleFinal.h5")
print("Model ready.")

# Ensure this URL matches your localtunnel output exactly
BASE_URL = "https://candle-daydream-myself.ngrok-free.dev"

# ... (segment_and_detect, preprocess, and draw_annotations functions remain the same) ...

def segment_and_detect(img_rgb):
    hsv = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions, boxes = [], []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > 500:
            crop = img_rgb[y:y+h, x:x+w]
            if crop.size > 0:
                regions.append(crop)
                boxes.append((x, y, w, h))
    return (regions, boxes) if regions else ([img_rgb], [])

def preprocess(img_rgb):
    img = cv2.resize(img_rgb, (IMG_SIZE, IMG_SIZE))
    return np.expand_dims(img / 255.0, axis=0)

def draw_annotations(img_rgb, boxes, predictions):
    output = img_rgb.copy()
    class_colors = {
        "Healthy": (46, 204, 113), "Apple Scab": (241, 196, 15),
        "Black Rot": (231, 76, 60), "Cedar Rust": (155, 89, 182),
    }
    for (x, y, w, h), (label, conf) in zip(boxes, predictions):
        color = class_colors.get(label, (255, 0, 0))
        text = f"{label} ({conf:.2f})"
        cv2.rectangle(output, (x, y), (x+w, y+h), color, 3)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.65, 2)
        label_y = max(y - 12, th + 12)
        cv2.rectangle(output, (x, label_y-th-8), (x+tw+8, label_y+4), color, -1)
        cv2.putText(output, text, (x+4, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255,255,255), 2)
    return output

def cleanup_old_images():
    while True:
        time.sleep(300)
        now = time.time()
        for fname in os.listdir(OUTPUT_DIR):
            fpath = os.path.join(OUTPUT_DIR, fname)
            if os.path.isfile(fpath) and now - os.path.getmtime(fpath) > 300:
                os.remove(fpath)

threading.Thread(target=cleanup_old_images, daemon=True).start()

@app.get("/")
def health():
    return {"status": "ok", "model": "Apple Leaf CNN"}

@app.post("/predict")
async def predict(leafImage: UploadFile = File(...)):
    contents = await leafImage.read()
    pil_img = Image.open(io.BytesIO(contents)).convert("RGB")
    img_rgb = np.array(pil_img)

    regions, boxes = segment_and_detect(img_rgb)
    all_preds = [model.predict(preprocess(r), verbose=0)[0] for r in regions]
    avg_pred = np.mean(all_preds, axis=0)
    
    region_predictions = [(DISPLAY_LABELS[int(np.argmax(p))], round(float(np.max(p)), 2)) for p in all_preds]
    annotated = draw_annotations(img_rgb, boxes, region_predictions)

    filename = f"{uuid.uuid4().hex}.jpg"
    save_path = os.path.join(OUTPUT_DIR, filename)
    cv2.imwrite(save_path, cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR))

    # ADDED TIMESTAMP to prevent FlutterFlow from caching broken/old images
    image_url = f"{BASE_URL.rstrip('/')}/outputs/{filename}?t={int(time.time())}"

    return {
        "predicted_class": DISPLAY_LABELS[int(np.argmax(avg_pred))],
        "confidence": round(float(np.max(avg_pred)) * 100, 2),
        "all_scores": {k: round(float(s), 4) for k, s in zip(SCORE_KEYS, avg_pred)},
        "regions_detected": len(regions),
        "annotated_image_url": image_url,
        "filename": filename,
    }

@app.delete("/delete-image")
async def delete_image(filename: str = Query(...)):
    file_path = os.path.join(OUTPUT_DIR, filename)
    if os.path.isfile(file_path):
        os.remove(file_path)
        return {"status": "deleted"}
    return {"status": "not_found"}