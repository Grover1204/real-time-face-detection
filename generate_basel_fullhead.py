import numpy as np
import open3d as o3d
from mediapipe.python.solutions.face_mesh_connections import FACEMESH_TESSELATION, FACEMESH_FACE_OVAL

# Dummy perfectly symmetrical template face points (canonical)
# Actually, let's load the template_face.obj we have
mesh_orig = o3d.io.read_triangle_mesh("models/template_face.obj")
vertices = np.asarray(mesh_orig.vertices)

back_vertex = np.mean(vertices, axis=0)
z_min = np.min(vertices[:, 2])
z_max = np.max(vertices[:, 2])
depth = z_max - z_min
# Move back vertex far behind the face to form the skull
back_vertex[2] = z_min - depth * 1.5
back_vertex[1] += depth * 0.3  # Crown of head

vertices = np.vstack((vertices, back_vertex))
back_idx = 468

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

for edge in FACEMESH_FACE_OVAL:
    v1, v2 = edge
    # Winding order checking (we just add both to ensure it connects and Open3D will fix normally or subdivide handles it)
    triangles.append((v1, v2, back_idx))

mesh = o3d.geometry.TriangleMesh()
mesh.vertices = o3d.utility.Vector3dVector(vertices)
mesh.triangles = o3d.utility.Vector3iVector(np.asarray(triangles, dtype=np.int32))
mesh.compute_vertex_normals()

# We save this as the basel_template.obj base structure
o3d.io.write_triangle_mesh("models/basel_template.obj", mesh)
print("Generated full-head models/basel_template.obj")
