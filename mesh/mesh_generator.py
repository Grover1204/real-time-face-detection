import open3d as o3d
import numpy as np
import os

def fit_basel_template(landmarks_3d, template_path="models/basel_template.obj"):
    """
    Fits the Basel Face Model (template) extending to a full head to the detected MediaPipe landmarks.
    Uses Kabsch / Umeyama algorithm on the 468 facial landmarks to compute rigid alignment,
    then applies it to the full skull template.
    """
    print("Loading 3D Morphable template (Full Head)...")
    
    # 1. Load the template mesh
    template_mesh = o3d.io.read_triangle_mesh(template_path)
    template_vertices = np.asarray(template_mesh.vertices)
    template_triangles = np.asarray(template_mesh.triangles)
    
    # 2. Extract corresponding frontal facial vertices (first 468) for alignment
    # Our generated 'models/basel_template.obj' has 468 facial landmarks matching 1:1, 
    # and additional vertices for the back of the head.
    frontal_template = template_vertices[:468]
    
    # 3. Calculate rigid alignment parameters (Kabsch / Umeyama) on frontal face ONLY
    print("Applying Rigid Alignment transforms...")
    mu_L = landmarks_3d.mean(axis=0)
    mu_T = frontal_template.mean(axis=0)
    
    L_centered = landmarks_3d - mu_L
    T_centered = frontal_template - mu_T
    
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
    
    # 4. Transform ALL template vertices (including back of head)
    aligned_template_vertices = scale * (template_vertices @ R.T) + t
    
    # 5. Build open3d model
    print("Building Open3D fitted mesh...")
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(aligned_template_vertices)
    mesh.triangles = o3d.utility.Vector3iVector(template_triangles)
    
    mesh.compute_vertex_normals()
    mesh.compute_triangle_normals()
    
    print(f"Base fitted mesh has {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
    
    print("Applying 1-iteration Loop subdivision...")
    dense_mesh = mesh.subdivide_loop(number_of_iterations=1)
    
    print(f"Dense mesh has {len(dense_mesh.vertices)} vertices and {len(dense_mesh.triangles)} triangles")
    
    # Rotate 180 degrees around X-axis so it is upright for Open3D viewer
    R_rot = dense_mesh.get_rotation_matrix_from_xyz((np.pi, 0, 0))
    dense_mesh.rotate(R_rot, center=(0, 0, 0))
    
    # Paint wireframe base blue
    dense_mesh.paint_uniform_color([0.2, 0.4, 0.8])
    
    return dense_mesh

def save_mesh(mesh, save_path):
    """
    Saves the Open3D mesh to the given path.
    """
    os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
    o3d.io.write_triangle_mesh(save_path, mesh)
    return save_path
