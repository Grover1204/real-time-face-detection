import cv2
import argparse
import sys
import imutils
from backend.detector import FaceDetector
from object_detector import ObjectDetector
from utils import FPS, draw_text

def get_args():
    parser = argparse.ArgumentParser(description="Real-time Webcam Detection System")
    parser.add_argument('--mode', type=str, default='face', choices=['face', 'object'],
                        help="Detection mode: 'face' for face detection, 'object' for object detection.")
    parser.add_argument('--camera', type=int, default=0,
                        help="Camera index (0 for default webcam).")
    return parser.parse_args()

def main():
    args = get_args()
    print(f"[INFO] Starting application in '{args.mode.upper()}' mode...")

    # Initialize the corresponding detector based on the selected mode
    if args.mode == 'face':
        detector = FaceDetector(confidence_threshold=0.5)
    else:
        detector = ObjectDetector(confidence_threshold=0.5)

    # Initialize the webcam
    print(f"[INFO] Initializing webcam at index {args.camera}...")
    cap = cv2.VideoCapture(args.camera)
    
    if not cap.isOpened():
        print("[ERROR] Could not open webcam. Make sure another application is not using it.")
        sys.exit(1)
        
    # Start the FPS tracker
    fps_tracker = FPS().start()

    print("[INFO] Ready! Press 'q' to exit.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to grab frame from webcam.")
            break

        # Resize the frame for a slight performance boost
        frame = imutils.resize(frame, width=800)

        # Process the frame based on the active mode
        if args.mode == 'face':
            detections = detector.detect_faces(frame)
            frame = detector.draw_faces(frame, detections)
        elif args.mode == 'object':
            detections = detector.detect_objects(frame)
            frame = detector.draw_objects(frame, detections)

        # Update and Display the FPS counter
        fps_tracker.update()
        current_fps = fps_tracker.fps()
        draw_text(frame, f"FPS: {current_fps:.1f}", 20, 30, color=(255, 255, 0), font_scale=0.7)
        
        # Display instructions for clean exit
        draw_text(frame, "Press 'q' to quit", 20, frame.shape[0] - 20, color=(0, 0, 255), font_scale=0.6)

        # Show the processed frame on the screen
        cv2.imshow("Real-Time Detection System", frame)

        # Wait for key press and exit cleanly if 'q' is pressed
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("[INFO] User requested shutdown. Exiting...")
            break

    # Clean up and release hardware resources
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Program closed smoothly.")

if __name__ == "__main__":
    main()
