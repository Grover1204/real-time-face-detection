import mediapipe as mp


class FaceDetector:
    def __init__(self, min_detection_confidence=0.5):
        """
        Initializes the MediaPipe Face Detection model.
        """
        self.mp_face_detection = mp.solutions.face_detection
        self.detector = self.mp_face_detection.FaceDetection(
            model_selection=1,  # 0 for short-range, 1 for full-range (up to 5 meters)
            min_detection_confidence=min_detection_confidence,
        )

    def detect(self, image_rgb, return_metrics=False):
        """
        Detects faces in an RGB image.
        Returns the bounding box of the most prominent face with a 20% margin: (xmin, ymin, width, height)
        If return_metrics is True, it returns (bbox, metrics_dict).
        """
        results = self.detector.process(image_rgb)

        metrics = {"num_faces": 0, "selected_index": None, "confidence": None}

        if not results.detections:
            if return_metrics:
                return None, metrics
            return None

        metrics["num_faces"] = len(results.detections)

        # Find the most prominent face (largest bounding box area)
        largest_area = 0
        best_bbox = None
        best_idx = 0
        best_conf = 0.0

        for i, detection in enumerate(results.detections):
            bboxC = detection.location_data.relative_bounding_box
            area = bboxC.width * bboxC.height
            if area > largest_area:
                largest_area = area
                best_bbox = bboxC
                best_idx = i
                if hasattr(detection, "score") and len(detection.score) > 0:
                    best_conf = detection.score[0]
                else:
                    best_conf = 1.0  # default fallback

        if best_bbox is None:
            if return_metrics:
                return None, metrics
            return None

        metrics["selected_index"] = best_idx
        metrics["confidence"] = best_conf

        ih, iw, _ = image_rgb.shape
        xmin = int(best_bbox.xmin * iw)
        ymin = int(best_bbox.ymin * ih)
        width = int(best_bbox.width * iw)
        height = int(best_bbox.height * ih)

        # Add a 20% margin to preserve chin and forehead
        margin_x = int(width * 0.20)
        margin_y = int(height * 0.20)

        xmin = max(0, xmin - margin_x)
        ymin = max(0, ymin - margin_y)

        # Final width and height shouldn't exceed image bounds
        xmax = min(iw, xmin + width + 2 * margin_x)
        ymax = min(ih, ymin + height + 2 * margin_y)

        final_width = xmax - xmin
        final_height = ymax - ymin

        bbox_result = (xmin, ymin, final_width, final_height)

        if return_metrics:
            return bbox_result, metrics
        return bbox_result

    def close(self):
        """
        Releases MediaPipe resources.
        """
        self.detector.close()
