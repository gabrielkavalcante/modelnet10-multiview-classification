from render_utils import render_modelnet_dataset


DATASET_PATH = "ModelNet10"
OUTPUT_PATH = "ModelNet10_36views_224x224"

IMAGE_WIDTH = 224
IMAGE_HEIGHT = 224

N_VIEWS = 36
VIEW_ANGLES_DEG = list(range(0, 360, 10))

ELEVATION_DEG = 30.0
CAMERA_DISTANCE = 3.8
ZOOM = 0.80

LIMIT_CLASSES = None
LIMIT_FILES_PER_SPLIT = None


def main():
    render_modelnet_dataset(
        dataset_path=DATASET_PATH,
        output_path=OUTPUT_PATH,
        n_views=N_VIEWS,
        view_angles=VIEW_ANGLES_DEG,
        image_width=IMAGE_WIDTH,
        image_height=IMAGE_HEIGHT,
        elevation_deg=ELEVATION_DEG,
        camera_distance=CAMERA_DISTANCE,
        zoom=ZOOM,
        limit_classes=LIMIT_CLASSES,
        limit_files_per_split=LIMIT_FILES_PER_SPLIT
    )


if __name__ == "__main__":
    main()