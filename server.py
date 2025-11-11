import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import FileResponse
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import io
from PIL import Image
import os

# NEW: Import the CORS middleware
from fastapi.middleware.cors import CORSMiddleware

# Import your existing Python modules
try:
    from models import yolo_model
    from facenet_recognition import recognize_face_and_get_text
    from ocr import read_text_from_frame
    print("All local modules imported successfully.")
except ImportError as e:
    print(f"Error importing local modules: {e}")
    print("Please ensure models.py, facenet_recognition.py, and ocr.py are in the same directory.")
    exit()

app = FastAPI()

# --- NEW: Add CORS Middleware ---
# This block tells the server to allow web browsers
# from any origin to make requests.
origins = ["*"]  # Allows all origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],  # Allows all methods (GET, POST, OPTIONS, etc.)
    allow_headers=["*"],  # Allows all headers
)
# --- End of new block ---


# --- Helper Function ---
def frame_from_b64(b64_string: str) -> np.ndarray:
    """Decodes a Base64 string into a CV2 Numpy array."""
    try:
        img_data = base64.b64decode(b64_string)
        pil_image = Image.open(io.BytesIO(img_data))
        frame_rgb = np.array(pil_image)
        frame_bgr = cv2.cvtColor(frame_rgb, cv2.COLOR_RGB2BGR)
        return frame_bgr
    except Exception as e:
        print(f"Error decoding base64 image: {e}")
        return None

# --- Pydantic Models for API ---
class ApiRequest(BaseModel):
    command: str
    image_b64: str

class ApiResponse(BaseModel):
    text: str

# --- Endpoint to serve your HTML file ---
@app.get("/")
async def get_index():
    """Serves the main index.html web app."""
    if not os.path.exists("index.html"):
        print("ERROR: index.html not found in this directory!")
        return {"error": "index.html not found"}, 404
    print("Serving index.html")
    return FileResponse("index.html")

# --- API Endpoint ---
@app.post("/api/analyze", response_model=ApiResponse)
async def analyze_frame(request: ApiRequest):
    """
    Main endpoint for all reactive, high-detail analysis.
    The mobile app sends a command and a frame here.
    """
    print(f"[SERVER] Received command: {request.command}")
    
    frame = frame_from_b64(request.image_b64)
    if frame is None:
        return ApiResponse(text="Sorry, I could not read the image.")

    response_text = "I'm not sure what you asked."
    try:
        if "who" in request.command:
            response_text = recognize_face_and_get_text(frame)
            print(f"[SERVER] Face recognition result: {response_text}")

        elif "read" in request.command:
            # --- THIS IS THE FIX ---
            response_text = read_text_from_frame(
                frame, 
                boxes=[], 
                use_cnn=False, 
                debug_dir="debug_images"  # <-- ADD THIS LINE
            )
            # --- END OF FIX ---

        elif "what" in request.command:
            results = yolo_model(frame)
            names = yolo_model.names
            detected_objects = set()
            for r in results:
                for box in r.boxes:
                    cls_id = int(box.cls[0])
                    label = names[cls_id]
                    detected_objects.add(label)
            
            if not detected_objects:
                response_text = "I don't see any objects."
            else:
                response_text = "I see: " + ", ".join(list(detected_objects))
        
        else:
            response_text = "Sorry, I didn't recognize that command."
    except Exception as e:
        print(f"[SERVER] Error during processing: {e}")
        response_text = "I encountered an error trying to analyze that."

    print(f"[SERVER] Sending response: {response_text}")
    return ApiResponse(text=response_text)

if __name__ == "__main__":
    print("--- Starting AI Server (and Web App) with CORS ---")
    print(f"Looking for index.html in: {os.getcwd()}")
    print("Open your browser to http://127.0.0.1:8000")
    uvicorn.run(app, host="0.0.0.0",port=8000)