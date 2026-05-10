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
```

To generate the 36-view dataset:

```
python scripts/render_36views.py
```

The original ModelNet10 folder must be available in the repository root or in the path defined inside the rendering scripts.

## Dataset verification

After rendering, the generated datasets can be checked with:

```
python scripts/verify_12views.py
python scripts/verify_36views.py
```

The verification scripts check whether each object folder contains the expected number of rendered views.

## Experiments

Six experiments were evaluated:

| Experiment | Views | Model | Epochs |
|---|---:|---|---:|
| E1 | 12 | CNN from scratch | 30 |
| E2 | 12 | MobileNetV2 pretrained | 30 |
| E3 | 12 | MobileNetV2 fine-tuning | 30 |
| E1' | 36 | CNN from scratch | 10 |
| E2' | 36 | MobileNetV2 pretrained | 10 |
| E3' | 36 | MobileNetV2 fine-tuning | 10 |

## Results

The best result was obtained by MobileNetV2 with partial fine-tuning using 12 views, achieving 91.96% accuracy and 91.82% macro F1-score.

The final metrics are available in:

```text
results/metrics_summary.csv
```

Additional result files are organized as follows:

```text
results/reports/              complete text reports
results/histories/            training histories
results/confusion_matrices/   confusion matrices
```

## Repository structure

```text
scripts/      Rendering and dataset verification scripts
notebooks/    Training and evaluation notebooks
results/      Metrics, histories, reports, and confusion matrices
figures/      Figures used in the paper
docs/         Additional documentation
```

## Requirements

Install the required packages with:

```
pip install -r requirements.txt
```

The experiments were developed in Python using Open3D, TensorFlow/Keras, NumPy, Pandas, Matplotlib, and Scikit-learn.

## Reproducibility

The dataset split was performed at the object level. The final prediction was computed by averaging the class probabilities obtained from all rendered views of the same object.

## Notes

The trained model files and rendered datasets are not stored in this repository due to file size. The repository provides the scripts, notebooks, histories, reports, and confusion matrices required to reproduce and inspect the experiments.
