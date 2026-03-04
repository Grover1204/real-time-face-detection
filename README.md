# Real-Time Webcam Face Detection

A computer vision application built with Python and OpenCV utilizing a Deep Neural Network (DNN) face detection model for real-time processing via the device webcam.

## Features
- **Real-Time Detection:** Employs OpenCV's DNN module for accurate face tracking.
- **Easy Setup:** Automatically downloads the required Caffe model and prototxt files on first run.
- **Performance:** Includes a live Frames Per Second (FPS) counter on the video feed.
- **Extensible:** Contains a modular structure to integrate further object detection features easily.

## Requirements
- Python 3.8+
- Webcam

## Installation

1. Clone this repository:
```bash
git clone https://github.com/Grover1204/real-time-face-detection.git
cd real-time-face-detection
```

2. Set up a virtual environment (optional but recommended):
```bash
python3 -m venv venv
source venv/bin/activate
```

3. Install the dependencies:
```bash
pip install -r requirements.txt
```

## Usage

Start the main application:
```bash
python main.py
```

### Controls:
- Ensure the webcam window is selected and press **`q`** to close the application cleanly.

## Project Structure
- `main.py` - Core execution script controlling video stream and mode operations.
- `face_detector.py` - Manages the OpenCV DNN Face Detection model operations.
- `object_detector.py` - A placeholder class template for adding object detection.
- `utils.py` - Helper utilities including FPS tracking and on-screen drawing tools.
- `models/` - Directory constructed for storing pre-trained weights (auto-downloaded).
