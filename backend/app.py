import os
import cv2
import numpy as np
import imutils
from fastapi import FastAPI, UploadFile, File, BackgroundTasks
from fastapi.responses import StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
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

stream_active = False
webcam_cap = None

def generate_webcam_frames():
    """Generator function that continuously yields webcam frames as JPEG images."""
    global stream_active, webcam_cap
    stream_active = True
    
    if webcam_cap is None or not webcam_cap.isOpened():
        webcam_cap = cv2.VideoCapture(0)
    
    try:
        while stream_active and webcam_cap is not None and webcam_cap.isOpened():
            ret, frame = webcam_cap.read()
            if not ret:
                break
                
            if frame.shape[1] > 800:
                frame = imutils.resize(frame, width=800)
                
            # Detect and draw
            detections = detector.detect_faces(frame)
            frame = detector.draw_faces(frame, detections)
            
            # Encode as JPEG
            _, buffer = cv2.imencode('.jpg', frame)
            frame_bytes = buffer.tobytes()
            
            # Generator for multipart stream
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
    finally:
        stream_active = False
        if webcam_cap is not None:
            webcam_cap.release()
            webcam_cap = None

@app.post("/stop-webcam")
async def stop_webcam_endpoint():
    """Endpoint to explicitly trigger the webcam hardware to release."""
    global stream_active, webcam_cap
    stream_active = False
    
    if webcam_cap is not None:
        webcam_cap.release()
        webcam_cap = None
        
    return {"status": "stopped"}

@app.get("/webcam-stream")
async def webcam_stream():
    """Streams live webcam frames with face detection."""
    return StreamingResponse(
        generate_webcam_frames(), 
        media_type="multipart/x-mixed-replace; boundary=frame"
    )

# Mount the static frontend directory at the root /
app.mount("/", StaticFiles(directory="frontend", html=True), name="frontend")
