import uvicorn
from fastapi import FastAPI, Request
from fastapi import WebSocket, WebSocketDisconnect
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel
import base64
import cv2
import numpy as np
import io
from PIL import Image
import os
from fastapi.middleware.cors import CORSMiddleware
import httpx
from datetime import datetime
import json as _json

# --- Create a folder for debug images ---
DEBUG_DIR = "debug_frames"
os.makedirs(DEBUG_DIR, exist_ok=True)
print(f"Saving incoming frames to: {os.path.abspath(DEBUG_DIR)}")
# ---------------------------------------------

# --- Set your ESP32 IP Address here ---
# This MUST be the IP of your ESP32-CAM on your WiFi network
ESP32_CAM_URL =  "http://10.248.176.19:81/stream"
# -----------------------------------------------

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

# --- Add CORS Middleware ---
# This allows the browser to make requests from different origins
origins = ["*"] 
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)
# --- End of CORS block ---


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

# --- Proxy Endpoint for Camera Stream ---
client = httpx.AsyncClient(timeout=None)

@app.get("/camera_stream")
async def get_camera_stream():
    """
    Proxies the ESP32-CAM's MJPEG stream to the browser with CORS headers.
    Guard against source stream errors so the app doesn't crash.
    """
    print(f"Client connected to camera stream. Proxying from: {ESP32_CAM_URL}")
    try:
        req = client.build_request("GET", ESP32_CAM_URL)
        r = await client.send(req, stream=True)

        proxy_headers = {
            "Access-Control-Allow-Origin": "*",
            "Cache-Control": "no-cache, no-store, must-revalidate",
            "Pragma": "no-cache",
            "Expires": "0",
        }
        ct = r.headers.get("content-type")
        if ct:
            proxy_headers["Content-Type"] = ct

        async def iter_bytes():
            try:
                async for chunk in r.aiter_bytes():
                    # yield only non-empty chunks
                    if chunk:
                        yield chunk
            except (httpx.ReadError, httpx.RemoteProtocolError) as e:
                # Source stream hiccup; log and end gracefully
                print(f"[STREAM] Source stream error: {e}")
            except Exception as e:
                print(f"[STREAM] Unexpected stream error: {e}")
            finally:
                try:
                    await r.aclose()
                except Exception:
                    pass

        return StreamingResponse(iter_bytes(), headers=proxy_headers, media_type=ct)

    except httpx.ConnectError as e:
        print(f"!!! CRITICAL ERROR: Could not connect to ESP32-CAM at {ESP32_CAM_URL}")
        print("Please check the IP address and that the ESP32-CAM is on the same network.")
        print(f"Error details: {e}")
        return {"error": "Could not connect to ESP32-CAM"}, 500
    except Exception as e:
        print(f"Error proxying stream setup: {e}")
        return {"error": str(e)}, 500

# --- API Endpoint for AI Analysis ---
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

    # --- Save the frame for debugging ---
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")[:-3]
        filename = f"{DEBUG_DIR}/{timestamp}_{request.command}.jpg"
        cv2.imwrite(filename, frame)
        print(f"Saved incoming frame to: {filename}")
    except Exception as e:
        print(f"Error saving frame: {e}")
    # ------------------------------------

    response_text = "I'm not sure what you asked."
    try:
        # --- COMMAND ROUTER ---
        if "who" in request.command:
            response_text = recognize_face_and_get_text(frame)
            print(f"[SERVER] Face recognition result: {response_text}")

        elif "read" in request.command:
            # Prefer Gemini OCR if configured (fall back to Tesseract/EasyOCR)
            text = read_text_from_frame(frame, boxes=[], use_cnn=False)
            if not text:
                response_text = "I couldn't detect any readable text."
            else:
                response_text = text.strip()

        elif "what" in request.command:
            # Use the powerful YOLO model for scene description
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


# ================== Voice WS (Vosk) ==================
# Reuse existing Vosk matching and grammar from voice_thread
try:
    from vosk import Model as _VoskModel, KaldiRecognizer as _KaldiRecognizer
    from voice_thread import _match_command as _vosk_match_command, VOCAB_GRAMMAR as _VOCAB_GRAMMAR
    _VOSK_AVAILABLE = True
except Exception as e:
    print(f"[VOICE-WS] Vosk not available: {e}")
    _VOSK_AVAILABLE = False

_vosk_model = None  # lazy init


@app.websocket("/ws/voice")
async def ws_voice(websocket: WebSocket):
    """WebSocket for streaming 16kHz mono PCM int16 audio to Vosk.
    Sends back matched commands as JSON: {type:"control", command:"pause|resume reading|stop reading|read|who|what"}
    """
    await websocket.accept()
    if not _VOSK_AVAILABLE:
        await websocket.send_json({"type": "error", "message": "Vosk not available on server"})
        await websocket.close()
        return

    global _vosk_model
    if _vosk_model is None:
        try:
            _vosk_model = _VoskModel(lang="en-us")
            print("[VOICE-WS] Vosk model initialized (en-us)")
        except Exception as e:
            await websocket.send_json({"type": "error", "message": f"Failed to load Vosk model: {e}"})
            await websocket.close()
            return

    # Grammar-constrained recognizer for faster, robust command matching
    rec = _KaldiRecognizer(_vosk_model, 16000, _json.dumps(list(_VOCAB_GRAMMAR)))

    try:
        while True:
            # Receive raw PCM int16 chunk
            data = await websocket.receive_bytes()
            if not data:
                continue
            if rec.AcceptWaveform(data):
                try:
                    result = _json.loads(rec.Result())
                except Exception:
                    result = {"text": ""}
                text = (result.get("text") or "").strip()
                if text:
                    print(f"[VOICE-WS] Final: {text}")
                    try:
                        cmd = _vosk_match_command(text)
                    except Exception:
                        cmd = None
                    if cmd:
                        await websocket.send_json({"type": "control", "command": cmd})
            else:
                # Optionally could send partials for UI; keeping quiet to reduce traffic
                pass
    except WebSocketDisconnect:
        print("[VOICE-WS] Client disconnected")
    except Exception as e:
        print(f"[VOICE-WS] Error: {e}")
        try:
            await websocket.send_json({"type": "error", "message": str(e)})
        except Exception:
            pass
        try:
            await websocket.close()
        except Exception:
            pass


if __name__ == "__main__":
    print("--- Starting AI Server (and Web App) with CORS ---")
    print(f"Looking for index.html in: {os.getcwd()}")
    print(f"Saving debug frames in: {os.path.abspath(DEBUG_DIR)}")
    print("Make sure to set 'ESP32_CAM_URL' in this file.")
    print("Run on your phone: http://[YOUR_LAPTOP_IP]:8000")
    uvicorn.run(app, host="0.0.0.0",port=8000)