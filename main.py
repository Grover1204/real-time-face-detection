import argparse
import os
import sys
import numpy as np
import cv2
import open3d as o3d

# Import local modules
from utils.image_utils import load_image, save_image, crop_image
from detectors.face_detector import FaceDetector
from alignment.face_alignment import FaceAligner
from reconstruction.reconstruct_face import get_3d_landmarks


def main():
    parser = argparse.ArgumentParser(
        description="3D Face Reconstruction Pipeline - Stage 3"
    )
    parser.add_argument(
        "--image", type=str, required=True, help="Path to the input image file"
    )

    args = parser.parse_args()
    image_path = args.image

    # ---------------------------------------------------------
    # Step 1: Image Input
    # ---------------------------------------------------------
    print("Loading image...")
    try:
        image_rgb = load_image(image_path)
    except Exception as e:
        print(f"Error loading image: {e}")
        sys.exit(1)

    # Define output paths
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)

    cropped_face_path = os.path.join(output_dir, f"{base_name}_cropped.jpg")
    aligned_face_path = os.path.join(output_dir, f"{base_name}_aligned.jpg")
    landmarks_face_path = os.path.join(output_dir, f"{base_name}_landmarks.jpg")

    # ---------------------------------------------------------
    # Stage 1: Face Detection
    # ---------------------------------------------------------
    print("Detecting face...")
    detector = FaceDetector()
    bbox = detector.detect(image_rgb)
    detector.close()

    if bbox is None:
        print("No face detected in the image.")
        sys.exit(1)

    print(f"Face detected at bounding box: {bbox}")

    print("Cropping face...")
    cropped_face = crop_image(image_rgb, bbox)

    print("Saving cropped image...")
    save_image(cropped_face, cropped_face_path)

    # ---------------------------------------------------------
    # Stage 2: Face Alignment
    # ---------------------------------------------------------
    print("Loading cropped face...")

    aligner = FaceAligner()
    aligned_face, debug_landmarks = aligner.align(cropped_face)
    aligner.close()

    if debug_landmarks is not None:
        save_image(debug_landmarks, landmarks_face_path)

    print("Saving aligned image...")
    if aligned_face is None:
        print("Alignment failed.")
        sys.exit(1)

    save_image(aligned_face, aligned_face_path)

    # ---------------------------------------------------------
    # Stage 3: 3D Geometry Extraction
    # ---------------------------------------------------------
    print("Loading aligned face image...")
    # (We already have aligned_face in memory)

    print("Running MediaPipe FaceMesh...")
    print("Extracting 468 landmarks...")
    print("Converting coordinates...")
    landmarks_3d = get_3d_landmarks(aligned_face)
    if landmarks_3d is None:
        print("Failed to extract 3D landmarks.")
        sys.exit(1)

    print("Saving NumPy landmark file...")
    np_output_path = os.path.join(output_dir, f"{base_name}_points.npy")
    np.save(np_output_path, landmarks_3d)

    print("Exporting point cloud file...")
    ply_output_path = os.path.join(output_dir, f"{base_name}_pointcloud.ply")
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(landmarks_3d)
    o3d.io.write_point_cloud(ply_output_path, pcd)

    # Save visualization
    vis_image = aligned_face.copy()
    for pt in landmarks_3d:
        cv2.circle(vis_image, (int(pt[0]), int(pt[1])), 1, (0, 255, 0), -1)

    vis_output_path = os.path.join(output_dir, f"{base_name}_points_visualization.jpg")
    save_image(vis_image, vis_output_path)

    # ---------------------------------------------------------
    # Stage 4: Mesh Generation and Visualization
    # ---------------------------------------------------------
    from mesh.mesh_generator import fit_basel_template, save_mesh
    from visualization.viewer import view_and_save_mesh

    print("Loading landmark points and generating 3D mesh...")

    dense_mesh = fit_basel_template(landmarks_3d)

    face_model_obj_path = os.path.join(output_dir, f"{base_name}_face_mesh.obj")
    face_model_ply_path = os.path.join(output_dir, f"{base_name}_face_mesh.ply")
    wireframe_preview_path = os.path.join(
        output_dir, f"{base_name}_wireframe_preview.png"
    )

    print("Saving dense face model...")
    save_mesh(dense_mesh, face_model_obj_path)
    save_mesh(dense_mesh, face_model_ply_path)

    view_and_save_mesh(dense_mesh, wireframe_preview_path)


if __name__ == "__main__":
    main()
