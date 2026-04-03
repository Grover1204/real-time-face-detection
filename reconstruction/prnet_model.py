import numpy as np
import mediapipe as mp


class PRNetModelApproximation:
    """
    Simulates a 3D geometry extraction model by wrapping MediaPipe's dense FaceMesh (468 vertices).
    """

    def __init__(self, static_image_mode=True, min_detection_confidence=0.5):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            static_image_mode=static_image_mode,
            max_num_faces=1,
            refine_landmarks=False,
            min_detection_confidence=min_detection_confidence,
        )

    def extract_landmarks(self, image_rgb):
        """
        Predicts dense 3D face vertices from the given aligned image.
        Returns:
            vertices: numpy array of shape (468, 3) containing x, y, z coordinates
        """
        results = self.face_mesh.process(image_rgb)

        if not results.multi_face_landmarks:
            print("Warning: No face landmarks detected for 3D reconstruction.")
            return None

        landmarks = results.multi_face_landmarks[0]
        h, w, _ = image_rgb.shape

        vertices = []
        for landmark in landmarks.landmark:
            # Scale x and y to image width and height
            x = landmark.x * w
            y = landmark.y * h
            # Keep z as relative depth per constraint
            z = landmark.z

            vertices.append([x, y, z])

        vertices = np.array(vertices, dtype=np.float32)
        return vertices

    def close(self):
        self.face_mesh.close()
