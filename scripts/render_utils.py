import glob
from pathlib import Path

import numpy as np
import open3d as o3d


def read_off(file_path):
    with open(file_path, "r", encoding="utf-8") as file:
        first_line = file.readline().strip()

        if first_line == "OFF":
            counts_line = file.readline().strip()
        elif first_line.startswith("OFF"):
            counts_line = first_line[3:].strip()
        else:
            raise ValueError(f"Invalid OFF file: {file_path}")

        while counts_line.startswith("#") or counts_line == "":
            counts_line = file.readline().strip()

        n_vertices, n_faces, _ = map(int, counts_line.split())

        vertices = []
        for _ in range(n_vertices):
            line = file.readline().strip()
            while line.startswith("#") or line == "":
                line = file.readline().strip()
            vertices.append(list(map(float, line.split())))

        faces = []
        for _ in range(n_faces):
            line = file.readline().strip()
            while line.startswith("#") or line == "":
                line = file.readline().strip()
            parts = list(map(int, line.split()))
            faces.append(parts[1:])

    return np.array(vertices, dtype=np.float64), faces


def triangulate_faces(faces):
    triangles = []

    for face in faces:
        if len(face) == 3:
            triangles.append(face)
        elif len(face) > 3:
            for i in range(1, len(face) - 1):
                triangles.append([face[0], face[i], face[i + 1]])

    return np.array(triangles, dtype=np.int32)


def normalize_vertices(vertices):
    min_bound = vertices.min(axis=0)
    max_bound = vertices.max(axis=0)

    center = (min_bound + max_bound) / 2.0
    vertices = vertices - center

    extent = max_bound - min_bound
    scale = np.max(extent)

    if scale > 0:
        vertices = vertices / scale

    return vertices


def create_mesh(vertices, triangles):
    mesh = o3d.geometry.TriangleMesh()
    mesh.vertices = o3d.utility.Vector3dVector(vertices)
    mesh.triangles = o3d.utility.Vector3iVector(triangles)

    mesh.compute_vertex_normals()
    mesh.paint_uniform_color([0.72, 0.72, 0.72])

    return mesh


def render_view(
    mesh,
    output_file,
    angle_deg,
    image_width,
    image_height,
    elevation_deg,
    camera_distance,
    zoom
):
    angle = np.deg2rad(angle_deg)
    elevation = np.deg2rad(elevation_deg)

    visualizer = o3d.visualization.Visualizer()
    visualizer.create_window(
        width=image_width,
        height=image_height,
        visible=False
    )

    visualizer.add_geometry(mesh)

    render_option = visualizer.get_render_option()
    render_option.background_color = np.asarray([1.0, 1.0, 1.0])
    render_option.mesh_show_wireframe = False
    render_option.light_on = True

    view_control = visualizer.get_view_control()

    bbox = mesh.get_axis_aligned_bounding_box()
    center = np.asarray(bbox.get_center())

    camera_x = center[0] + camera_distance * np.cos(elevation) * np.cos(angle)
    camera_y = center[1] + camera_distance * np.cos(elevation) * np.sin(angle)
    camera_z = center[2] + camera_distance * np.sin(elevation)

    camera_position = np.array([camera_x, camera_y, camera_z])

    front = camera_position - center
    front = front / np.linalg.norm(front)

    up = np.array([0.0, 0.0, 1.0])

    view_control.set_lookat(center)
    view_control.set_front(front)
    view_control.set_up(up)
    view_control.set_zoom(zoom)

    visualizer.poll_events()
    visualizer.update_renderer()
    visualizer.capture_screen_image(str(output_file), do_render=True)
    visualizer.destroy_window()


def get_classes(dataset_path, limit_classes=None):
    classes = [
        item.name
        for item in Path(dataset_path).iterdir()
        if item.is_dir()
    ]

    classes = sorted(classes)

    if limit_classes is not None:
        classes = [class_name for class_name in classes if class_name in limit_classes]

    return classes


def process_off_file(
    file_path,
    output_folder,
    view_angles,
    image_width,
    image_height,
    elevation_deg,
    camera_distance,
    zoom
):
    vertices, faces = read_off(file_path)
    vertices = normalize_vertices(vertices)
    triangles = triangulate_faces(faces)

    mesh = create_mesh(vertices, triangles)

    output_folder.mkdir(parents=True, exist_ok=True)

    for angle_deg in view_angles:
        output_file = output_folder / f"view_{angle_deg:03d}.png"

        render_view(
            mesh=mesh,
            output_file=output_file,
            angle_deg=angle_deg,
            image_width=image_width,
            image_height=image_height,
            elevation_deg=elevation_deg,
            camera_distance=camera_distance,
            zoom=zoom
        )


def render_modelnet_dataset(
    dataset_path,
    output_path,
    n_views,
    view_angles,
    image_width=224,
    image_height=224,
    elevation_deg=30.0,
    camera_distance=3.8,
    zoom=0.80,
    limit_classes=None,
    limit_files_per_split=None
):
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset path not found: {dataset_path}")

    output_path.mkdir(parents=True, exist_ok=True)

    classes = get_classes(dataset_path, limit_classes=limit_classes)

    print("Starting ModelNet10 rendering")
    print(f"Dataset path: {dataset_path}")
    print(f"Output path: {output_path}")
    print(f"Number of classes: {len(classes)}")
    print(f"Classes: {classes}")
    print(f"Views per object: {n_views}")
    print(f"View angles: {view_angles}")
    print(f"Image size: {image_width}x{image_height}")
    print(f"Camera elevation: {elevation_deg}")
    print(f"Camera distance: {camera_distance}")
    print(f"Zoom: {zoom}")

    total_files = 0

    for class_name in classes:
        for split in ["train", "test"]:
            pattern = dataset_path / class_name / split / "*.off"
            files = sorted(glob.glob(str(pattern)))

            if limit_files_per_split is not None:
                files = files[:limit_files_per_split]

            total_files += len(files)

    print(f"Total OFF files to process: {total_files}")

    processed_files = 0

    for class_name in classes:
        for split in ["train", "test"]:
            pattern = dataset_path / class_name / split / "*.off"
            files = sorted(glob.glob(str(pattern)))

            if limit_files_per_split is not None:
                files = files[:limit_files_per_split]

            print(f"Processing class: {class_name}, split: {split}, files: {len(files)}")

            for file_path in files:
                file_path = Path(file_path)
                file_stem = file_path.stem

                output_folder = output_path / split / class_name / file_stem

                processed_files += 1
                print(f"Processing file {processed_files}/{total_files}: {class_name}/{split}/{file_stem}")

                process_off_file(
                    file_path=file_path,
                    output_folder=output_folder,
                    view_angles=view_angles,
                    image_width=image_width,
                    image_height=image_height,
                    elevation_deg=elevation_deg,
                    camera_distance=camera_distance,
                    zoom=zoom
                )

    print("Rendering completed successfully")
    print(f"Processed OFF files: {processed_files}")
    print(f"Rendered dataset saved at: {output_path}")