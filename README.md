# Real-Time Face Detection AI

![AI Vision Demo](https://img.shields.io/badge/Status-Active-brightgreen)
![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![FastAPI](https://img.shields.io/badge/Backend-FastAPI-teal)

A sleek, enterprise-grade AI Vision dashboard that runs real-time computer vision right in your browser. Powered by OpenCV's Deep Neural Networks (DNN) and wrapped in a beautiful, modern "Apple Liquid Glass" UI.

## Features
- **🎥 Web Live Demo:** Stream directly from your webcam with instant bounding-box renders.
- **🖼️ Static Image Analysis:** Upload high-res images to detect multiple faces instantly.
- **🎬 Video Processing:** Upload MP4s to batch-process frames and generate output videos.
- **⚡ Auto-Setup:** Automatically downloads the required Caffe models on first run.

## Live Demo
Check out the fully working, live version of this project running directly in your browser:
**[View Live Demo on Hugging Face](https://huggingface.co/spaces/Grover1204/webcame-face-detection)**

---

## How to Run Locally



You only need Python installed to run the full dashboard!

1. **Clone the repository:**
```bash
git clone https://github.com/Grover1204/real-time-face-detection.git
cd real-time-face-detection
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Start the server:**
```bash
uvicorn backend.app:app --reload
```

4. **Open the studio:**
Visit `http://localhost:8000` in your web browser.


## Requirements & Notes
- Python 3.8 or higher
- A working webcam for real-time detection
- Tested on macOS and Linux (should also work on Windows)
- No GPU required (CPU-based OpenCV DNN)

> On first run, required Caffe models will be downloaded automatically.

---

## How to Deploy (Make it Live)
To make this live on the internet so anyone can use it, you need to host the Python backend. Some great free/cheap options:

### Option 1: HuggingFace Spaces (Currently Deployed Here)
1. Hosted on a Docker environment.
2. Built automatically via the included `Dockerfile`.
3. Runs the FastAPI and OpenCV neural network remotely!

### Option 2: Render.com
1. Sign up at [Render](https://render.com).
2. Create a new **Web Service**.
3. Connect this GitHub repository.
4. Set the Build Command to: `pip install -r requirements.txt`
5. Set the Start Command to: `uvicorn backend.app:app --host 0.0.0.0 --port $PORT`

---

## Author
**Rahul Grover**
- GitHub: [@Grover1204](https://github.com/Grover1204)
- Email: rahulgroveryk@gmail.com
