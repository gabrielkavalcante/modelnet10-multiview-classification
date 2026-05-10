# modelnet10-multiview-classification
Code for multi-view 2D projection and classification of 3D objects from ModelNet10.


```markdown
# ModelNet10 Multi-view Classification

This repository contains the code used for multi-view 2D projection and classification of 3D objects from the ModelNet10 dataset.

The original 3D objects were rendered into 2D images and classified using convolutional neural networks. The experiments compare a CNN trained from scratch, a pretrained MobileNetV2, and a MobileNetV2 with partial fine-tuning.

## Dataset

The experiments use the ModelNet10 dataset. The original `.off` files were rendered into two multi-view datasets:

- 12 views per object, with 30-degree angular intervals;
- 36 views per object, with 10-degree angular intervals.

All rendered images were generated with resolution 224x224, white background, gray object color, orbital camera, 30-degree elevation, camera distance of 3.8, and zoom of 0.80.

The rendered datasets are not included in this repository due to file size. They can be regenerated using the scripts in the `scripts/` folder.

## Rendering

To generate the 12-view dataset:

```
python scripts/render_12views.py

To generate the 36-view dataset:

python scripts/render_36views.py

The original ModelNet10 folder must be available in the repository root or in the path defined inside the rendering scripts.

Dataset verification

After rendering, the generated datasets can be checked with:

```
python scripts/verify_12views.py
python scripts/verify_36views.py
```
