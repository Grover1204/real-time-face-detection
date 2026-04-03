import cv2
import numpy as np
import mediapipe as mp


class FaceAligner:
    def __init__(
        self, static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
    ):
        """
        Initializes the MediaPipe FaceMesh for facial landmarks and alignment.
        """
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=max_num_faces,
            refine_landmarks=True,
            min_detection_confidence=min_detection_confidence,
        )

        # MediaPipe landmark indices
        self.LEFT_EYE_INDICES = [33, 133]  # corners of left eye
        self.RIGHT_EYE_INDICES = [362, 263]  # corners of right eye
        self.NOSE_TIP_INDEX = 1

    def align(self, image_rgb, return_metrics=False):
        """
        Aligns the face so that the eyes are horizontal and the face is centered.
        Returns (aligned_image, debug_image) or (image_rgb, None) if no face detected.
        If return_metrics is True, it returns (aligned_image, debug_image, metrics_dict).
        """

        results = self.face_mesh.process(image_rgb)

        metrics = {"left_eye": None, "right_eye": None, "rotation_angle": None}

        if not results.multi_face_landmarks:
            if return_metrics:
                return image_rgb, None, metrics
            return image_rgb, None

        landmarks = results.multi_face_landmarks[0]
        h, w, _ = image_rgb.shape

        # Extract features
        left_eye_pts = [
            np.array([landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h])
            for idx in self.LEFT_EYE_INDICES
        ]
        right_eye_pts = [
            np.array([landmarks.landmark[idx].x * w, landmarks.landmark[idx].y * h])
            for idx in self.RIGHT_EYE_INDICES
        ]
        nose_pt = np.array(
            [
                landmarks.landmark[self.NOSE_TIP_INDEX].x * w,
                landmarks.landmark[self.NOSE_TIP_INDEX].y * h,
            ]
        )

        # Calculate eye centers for rotation angle
        left_eye_center = np.mean(left_eye_pts, axis=0)
        right_eye_center = np.mean(right_eye_pts, axis=0)

        metrics["left_eye"] = (int(left_eye_center[0]), int(left_eye_center[1]))
        metrics["right_eye"] = (int(right_eye_center[0]), int(right_eye_center[1]))

        # Calculate angle to make eyes exactly horizontal
        dY = right_eye_center[1] - left_eye_center[1]
        dX = right_eye_center[0] - left_eye_center[0]
        angle = np.degrees(np.arctan2(dY, dX))

        metrics["rotation_angle"] = angle

        # Get rotation matrix pivoting on the nose
        nose_center = (int(nose_pt[0]), int(nose_pt[1]))
        M = cv2.getRotationMatrix2D(nose_center, angle, 1.0)

        # Add translation translation to perfectly center the face (put nose at image center)
        M[0, 2] += (w // 2) - nose_center[0]
        M[1, 2] += (h // 2) - nose_center[1]

        # Perform rotation and centering translation
        aligned_image = cv2.warpAffine(image_rgb, M, (w, h), flags=cv2.INTER_CUBIC)

        # Create debug visualization
        debug_image = image_rgb.copy()

        # Draw all landmarks
        for lm in landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            cv2.circle(debug_image, (x, y), 1, (0, 255, 0), -1)

        # Highlight eye centers and connecting line
        cv2.circle(debug_image, metrics["left_eye"], 4, (255, 0, 0), -1)
        cv2.circle(debug_image, metrics["right_eye"], 4, (255, 0, 0), -1)
        cv2.line(
            debug_image, metrics["left_eye"], metrics["right_eye"], (255, 255, 0), 2
        )

        if return_metrics:
            return aligned_image, debug_image, metrics
        return aligned_image, debug_image

    def close(self):
        self.face_mesh.close()
