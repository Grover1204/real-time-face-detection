import cv2
import os
import urllib.request
import numpy as np
import ssl

ssl._create_default_https_context = ssl._create_unverified_context

class FaceDetector:
    """
    DNN-based Face Detector using OpenCV's deep learning module.
    Automatically downloads the required Caffe model files if missing.
    """
    def __init__(self, confidence_threshold=0.5):
        self.confidence_threshold = confidence_threshold
        
        # Resolve paths dynamically relative to this file
        base_dir = os.path.dirname(os.path.abspath(__file__))
        self.models_dir = os.path.join(base_dir, "models")
        self.modelFile = os.path.join(self.models_dir, "res10_300x300_ssd_iter_140000.caffemodel")
        self.configFile = os.path.join(self.models_dir, "deploy.prototxt")
        
        self._ensure_models_exist()
        self.load_model()

    def _ensure_models_exist(self):
        """Downloads the pre-trained Caffe models natively if not present in the models/ dir."""
        os.makedirs(self.models_dir, exist_ok=True)
        
        if not os.path.exists(self.modelFile):
            print("[INFO] Downloading DNN Face Detector weights...")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv_3rdparty/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel",
                self.modelFile
            )
            
        if not os.path.exists(self.configFile):
            print("[INFO] Downloading DNN Config file...")
            urllib.request.urlretrieve(
                "https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt",
                self.configFile
            )

    def load_model(self):
        """Loads the pre-trained DNN face detection model."""
        print("[INFO] Loading face detection model...")
        self.net = cv2.dnn.readNetFromCaffe(self.configFile, self.modelFile)
        print("[INFO] Face detection model loaded successfully.")

    def detect_faces(self, frame):
        """
        Runs the face detection model over an image frame.
        Returns a list of detections: [ (x, y, w, h, confidence), ... ]
        """
        h, w = frame.shape[:2]
        
        # Preprocess the image: blobFromImage natively handles scaling the dimensions
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), (104.0, 177.0, 123.0))
        self.net.setInput(blob)
        detections = self.net.forward()
        
        detected_faces = []
        # Loop over the detections
        for i in range(0, detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            # Filter out weak detections
            if confidence > self.confidence_threshold:
                # Compute the (x, y)-coordinates of the bounding box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                # Ensure the bounding boxes fall within the dimensions of the frame
                startX = max(0, startX)
                startY = max(0, startY)
                endX = min(w - 1, endX)
                endY = min(h - 1, endY)
                
                width = endX - startX
                height = endY - startY
                
                if width > 0 and height > 0:
                    detected_faces.append((startX, startY, width, height, confidence))
                    
        return detected_faces

    def draw_faces(self, frame, detections):
        """Draws bounding boxes and confidence scores on the given frame."""
        for (x, y, w, h, confidence) in detections:
            # Draw the bounding box
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Format and draw the confidence text above the box
            text = f"Face Detected: {confidence * 100:.1f}%"
            text_y = y - 10 if y - 10 > 10 else y + 20
            
            cv2.putText(frame, text, (x, text_y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        return frame
