import open3d as o3d
import numpy as np


def view_and_save_mesh(mesh, save_screenshot_path):
    """
    Opens an interactive Open3D viewer window to display the dense wireframe 3D face mesh.
    Saves a screenshot to the provided path.
    """
    print("Opening 3D viewer...")

    vis = o3d.visualization.Visualizer()
    vis.create_window(window_name="3D Face Wireframe Viewer", width=1024, height=768)

    # Setup rendering options
    opt = vis.get_render_option()
    opt.background_color = np.asarray(
        [0.0, 0.0, 0.0]
    )  # Black background per requirements
    opt.mesh_show_wireframe = True  # Wireframe rendering per requirements
    opt.mesh_show_back_face = True

    vis.add_geometry(mesh)

    # Center camera on the geometry
    vis.poll_events()
    vis.update_renderer()

    # Capture screen image
    vis.capture_screen_image(save_screenshot_path, do_render=True)
    print(f"Saved wireframe preview to {save_screenshot_path}")
    print("Close the Open3D viewer window to continue.")

    # Run viewer (blocks until user closes the window)
    vis.run()
    vis.destroy_window()
