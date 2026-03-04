import cv2
import time

class FPS:
    """
    A simple frames per second tracker.
    """
    def __init__(self):
        self._start_time = None
        self._num_frames = 0
        self._current_fps = 0.0

    def start(self):
        """Start the fps timer."""
        self._start_time = time.time()
        return self

    def update(self):
        """Update the frame count and calculate FPS."""
        self._num_frames += 1
        elapsed = time.time() - self._start_time
        if elapsed > 0:
            self._current_fps = self._num_frames / elapsed

    def fps(self):
        """Return the current frames per second."""
        return self._current_fps

def draw_text(frame, text, x, y, color=(0, 255, 0), thickness=2, font_scale=0.6):
    """
    Helper function to draw text on an OpenCV frame.
    """
    cv2.putText(frame, text, (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness)
