# 3D Face Reconstruction Pipeline

A modular Python pipeline that reconstructs a 3D face mesh from a single input image.

## Overview

This project implements a complete production-quality pipeline for 3D face reconstruction using Python. It sequentially processes an input image performing:

1. Face Detection (MediaPipe)
2. Face Alignment
3. 3D Face Reconstruction (Dense mesh approximation)
4. Mesh Generation (OBJ export)
5. Interactive 3D Visualization (Open3D)

## Installation

1. Clone or download this project.
2. (Optional but recommended) Create a virtual environment using `python -m venv venv`.
3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run the pipeline using the `main.py` entry point:

```bash
python main.py --image <path_to_input_image>
```

Example:

```bash
python main.py --image sample_face.jpg
```

This will run the full reconstruction pipeline, save the resulting mesh to `outputs/face.obj`, and automatically open an interactive 3D viewer.
