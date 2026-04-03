from .prnet_model import PRNetModelApproximation
import open3d as o3d
import numpy as np


def get_3d_landmarks(aligned_image_rgb):
    """
    Initializes the model, runs 3D reconstruction on the aligned image, and returns the 468 vertices.
    """
    model = PRNetModelApproximation()
    vertices = model.extract_landmarks(aligned_image_rgb)
    model.close()

    return vertices


def fit_template_to_landmarks(landmarks_path, template_path="models/template_face.obj"):
    """
    Loads a 3D face template and deforms it to match the detected 3D landmarks (non-rigid alignment).
    """
    # 1. Load the template mesh
    template_mesh = o3d.io.read_triangle_mesh(template_path)
    template_vertices = np.asarray(template_mesh.vertices)
    template_triangles = np.asarray(template_mesh.triangles)

    # 2. Load the MediaPipe landmarks from Stage 3
    landmarks_3d = np.load(landmarks_path)

    # 3. Rigid alignment parameters (Kabsch / Umeyama algorithm)
    mu_L = landmarks_3d.mean(axis=0)
    mu_T = template_vertices.mean(axis=0)

    L_centered = landmarks_3d - mu_L
    T_centered = template_vertices - mu_T

    # Compute scale based on RMS distance
    rms_L = np.sqrt(np.mean(np.sum(L_centered**2, axis=1)))
    rms_T = np.sqrt(np.mean(np.sum(T_centered**2, axis=1)))
    scale = rms_L / rms_T if rms_T > 0 else 1.0

    # Compute rotation using SVD
    H = T_centered.T @ L_centered
    U, S, Vt = np.linalg.svd(H)
    R = Vt.T @ U.T

    # Handle reflection case
    if np.linalg.det(R) < 0:
        Vt[2, :] *= -1
        R = Vt.T @ U.T

    t = mu_L - scale * (mu_T @ R.T)

    # Apply rigid transformation to all template vertices
    aligned_template = scale * (template_vertices @ R.T) + t

    # 4. Mesh Deformation (Non-Rigid Warping)
    # The template face model and the MediaPipe landmarks share exactly 468 corresponding vertices.
    # To lock the face exactly to the image features while keeping realistic 3D depth,
    # we use the image-accurate X,Y coordinates from the landmarks and
    # the scaled real-volume Z coordinates from the canonical template.
    deformed_vertices = np.copy(landmarks_3d)
    deformed_vertices[:, 2] = aligned_template[:, 2]

    # Create final deformed base mesh
    fitted_mesh = o3d.geometry.TriangleMesh()
    fitted_mesh.vertices = o3d.utility.Vector3dVector(deformed_vertices)
    fitted_mesh.triangles = o3d.utility.Vector3iVector(template_triangles)
    fitted_mesh.compute_vertex_normals()
    fitted_mesh.compute_triangle_normals()

    verts = np.asarray(fitted_mesh.vertices)
    print("vertices.shape:", verts.shape)
    print("triangles.shape:", np.asarray(fitted_mesh.triangles).shape)
    print("vertices.min(axis=0):", verts.min(axis=0))
    print("vertices.max(axis=0):", verts.max(axis=0))

    return fitted_mesh
