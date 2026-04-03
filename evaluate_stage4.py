import os
import glob
import numpy as np

from utils.image_utils import load_image, crop_image
from detectors.face_detector import FaceDetector
from alignment.face_alignment import FaceAligner
from reconstruction.reconstruct_face import get_3d_landmarks
from mesh.mesh_generator import fit_basel_template, save_mesh


def run_stage4_tests():
    test_dir = (
        "/Users/grover/Documents/3d model/face_3d_reconstruction/test case images"
    )
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    # Get all jpg/jpeg files
    image_paths = glob.glob(os.path.join(test_dir, "*.jpg")) + glob.glob(
        os.path.join(test_dir, "*.jpeg")
    )
    image_paths = sorted(image_paths)

    total_images = len(image_paths)
    successful_meshes = 0
    failed_meshes = 0

    detector = FaceDetector()
    aligner = FaceAligner()

    for img_path in image_paths:
        img_name = os.path.basename(img_path)
        base_name = os.path.splitext(img_name)[0]
        print(f"\n{'='*50}")
        print(f"Processing: {img_name}")

        try:
            image_rgb = load_image(img_path)

            # Stage 1
            bbox = detector.detect(image_rgb)
            if bbox is None:
                print("Failed: No face detected.")
                failed_meshes += 1
                continue
            cropped_face = crop_image(image_rgb, bbox)

            # Stage 2
            aligned_face, debug_landmarks = aligner.align(cropped_face)
            if debug_landmarks is None:
                print("Failed: Alignment failed.")
                failed_meshes += 1
                continue

            # Stage 3
            landmarks_3d = get_3d_landmarks(aligned_face)
            if landmarks_3d is None or landmarks_3d.shape != (468, 3):
                print("Failed: Landmark extraction failed.")
                failed_meshes += 1
                continue

            # Stage 4
            dense_mesh = fit_basel_template(landmarks_3d)
            verts = np.asarray(dense_mesh.vertices)
            print(
                f"Mesh generated successfully! Vertices: {verts.shape[0]}, Triangles: {np.asarray(dense_mesh.triangles).shape[0]}"
            )

            # Save artifacts
            obj_path = os.path.join(output_dir, f"{base_name}_test_face_mesh.obj")
            save_mesh(dense_mesh, obj_path)
            successful_meshes += 1

        except Exception as e:
            print(f"Pipeline crashed on {img_name}: {e}")
            failed_meshes += 1

    detector.close()
    aligner.close()

    report_path = "stage4_test_report.txt"
    with open(report_path, "w") as f:
        f.write("Stage 4 Basel Template Mesh Test Summary\n\n")
        f.write(f"Total Images: {total_images}\n")
        f.write(f"Successful Meshes Generated: {successful_meshes}\n")
        f.write(f"Failed Pipelines: {failed_meshes}\n")

    print(f"\nSaved test report to {report_path}")


if __name__ == "__main__":
    run_stage4_tests()
