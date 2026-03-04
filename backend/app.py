import os
import cv2
import numpy as np
import imutils
from fastapi import FastAPI, UploadFile, File, BackgroundTasks, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import base64
from backend.detector import FaceDetector

app = FastAPI(title="Real-Time Face Detection AI Demo")

# Allow CORS for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the face detector
detector = FaceDetector(confidence_threshold=0.6)

# Ensure uploads directory exists
UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Function to clear old uploads (runs as background task)
def cleanup_file(filepath: str):
    if os.path.exists(filepath):
        os.remove(filepath)

# --- ENDPOINTS ---

@app.post("/detect-image")
async def detect_image(file: UploadFile = File(...)):
    """Accepts an image, detects faces, and returns the annotated image."""
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    if frame is None:
        return {"error": "Invalid image format"}

    # Resize image to prevent high-res aliasing artifacts forming false faces
    if frame.shape[1] > 800:
        frame = imutils.resize(frame, width=800)

    # Process frame
    detections = detector.detect_faces(frame)
    frame = detector.draw_faces(frame, detections)

    # Encode back to JPEG
    _, encoded_image = cv2.imencode('.jpg', frame)
    
    return StreamingResponse(
        iter([encoded_image.tobytes()]), 
        media_type="image/jpeg"
    )

@app.post("/detect-video")
async def detect_video(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    """Accepts a video file, processes it, and returns the annotated video."""
    input_path = os.path.join(UPLOAD_DIR, f"input_{file.filename}")
    output_path = os.path.join(UPLOAD_DIR, f"output_{file.filename}")
    
    # Save the uploaded video to disk
    with open(input_path, "wb") as f:
        f.write(await file.read())
        
    cap = cv2.VideoCapture(input_path)
    
    if not cap.isOpened():
        return {"error": "Could not open video file"}
        
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Calculate new width and height to maintain aspect ratio and prevent aliasing noise
    if width > 800:
        ratio = 800.0 / width
        width = 800
        height = int(height * ratio)
    
    # We must use mp4v for web browsers, or x264 if we want wide compatibility
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        if frame.shape[1] > 800:
            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)

        detections = detector.detect_faces(frame)
        frame = detector.draw_faces(frame, detections)
        out.write(frame)
        
    cap.release()
    out.release()
    
    # Add files to cleanup task later, we return the file immediately
    background_tasks.add_task(cleanup_file, input_path)
    
    return FileResponse(output_path, media_type="video/mp4", filename=f"processed_{file.filename}")

@app.websocket("/ws/webcam")
async def websocket_endpoint(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            # Receive base64 frame from client
            data = await websocket.receive_text()
            
            # Remove the "data:image/jpeg;base64," header if present
            if "," in data:
                data = data.split(",")[1]
                
            # Decode base64 to numpy array
            img_data = base64.b64decode(data)
            nparr = np.frombuffer(img_data, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            
            if frame is None:
                continue

            # Ensure optimal size for detection
            if frame.shape[1] > 800:
                frame = imutils.resize(frame, width=800)

            # Detect and draw
            detections = detector.detect_faces(frame)
            frame = detector.draw_faces(frame, detections)

            # Encode back to JPEG and to base64
            _, buffer = cv2.imencode('.jpg', frame)
            encoded_img = base64.b64encode(buffer).decode('utf-8')
            
            # Send back to client
            await websocket.send_text(f"data:image/jpeg;base64,{encoded_img}")
            
    except WebSocketDisconnect:
        print("Client disconnected from webcam stream")
    except Exception as e:
        print(f"WebSocket error: {str(e)}")

# Mount the static frontend directory at the root /
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
