import numpy as np
import open3d as o3d
from mediapipe.python.solutions.face_mesh_connections import (
    FACEMESH_TESSELATION,
    FACEMESH_FACE_OVAL,
)

landmarks = np.load("outputs/1_points.npy")

# Normalize
vertices = landmarks - np.mean(landmarks, axis=0)
scale = np.max(np.linalg.norm(vertices, axis=1))
vertices = vertices / scale * 100.0

# Add back-of-head vertex
back_vertex = np.mean(vertices, axis=0)

# The face points forwards (Z is usually positive towards viewer).
# Let's check min/max Z
z_min = np.min(vertices[:, 2])
z_max = np.max(vertices[:, 2])
# Depth of face
depth = z_max - z_min
# Move back vertex far behind the face
back_vertex[2] = z_min - depth * 1.5
back_vertex[1] += depth * 0.2  # slightly up for the crown of head

vertices = np.vstack((vertices, back_vertex))
back_idx = 468

# Build base triangles
adj = {i: set() for i in range(468)}
for edge in FACEMESH_TESSELATION:
    if edge[0] < 468 and edge[1] < 468:
        adj[edge[0]].add(edge[1])
        adj[edge[1]].add(edge[0])

triangles = set()
for i in range(468):
    for j in adj[i]:
        if j > i:
            for k in adj[j]:
                if k > j and i in adj[k]:
                    triangles.add((i, j, k))
triangles = list(triangles)

# Get oval edges and create triangles linking to back_idx
for edge in FACEMESH_FACE_OVAL:
    v1, v2 = edge
    triangles.append((v1, v2, back_idx))

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))

mesh.compute_vertex_normals()
mesh.compute_triangle_normals()

dense_mesh = mesh.subdivide_loop(number_of_iterations=2)

o3d.io.write_triangle_mesh("outputs/test_full_head.obj", dense_mesh)
print("Saved outputs/test_full_head.obj")
