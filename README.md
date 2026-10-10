# computer vision

I use these projects to work through image representations, classification, retrieval and reconstruction. The sequence starts with filters and local features, then compares CNNs, skip connections and patch attention. Each notebook contains executable experiments, figures and a description of what its metrics measure.

## Projects in reading order

| Order | Project |
| --- | --- |
| 01 | [Image filters and frequency content](notebooks/01_filters_and_frequency.ipynb) |
| 02 | [SIFT matching and geometric verification](notebooks/02_sift_and_geometry.ipynb) |
| 03 | [Object recognition with visual words](notebooks/03_visual_words.ipynb) |
| 04 | [Dense and convolutional digit classifiers](notebooks/04_mlp_and_lenet.ipynb) |
| 05 | [Augmentation, distribution shift and calibration](notebooks/05_augmentation_and_calibration.ipynb) |
| 06 | [Hand-sign recognition with multi-scale CNNs](notebooks/06_signs_and_inception.ipynb) |
| 07 | [Transfer to shifted images with a frozen encoder](notebooks/07_transfer_and_frozen_features.ipynb) |
| 08 | [Segmenting synthetic cell-shaped objects](notebooks/08_cell_segmentation.ipynb) |
| 09 | [Learning an image retrieval embedding](notebooks/09_metric_learning.ipynb) |
| 10 | [Denoising images with a convolutional autoencoder](notebooks/10_denoising_autoencoder.ipynb) |
| 11 | [Residual super-resolution on image patches](notebooks/11_super_resolution.ipynb) |
| 12 | [Input-gradient sensitivity and FGSM](notebooks/12_adversarial_sensitivity.ipynb) |
| 13 | [Patch attention on small images](notebooks/13_patch_transformer.ipynb) |

The early notebooks establish useful baselines. Later projects add new model assumptions and compare them against references where a common task allows it. Read each notebook's data and evaluation notes before comparing its scores with another project.

## Run locally

Use Python 3.10 or newer and a matched PyTorch and torchvision installation. For CPU-only installation, install those two packages from PyTorch's CPU wheel index first.

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
python -m pip install -e '.[dev]'
python scripts/download_data.py signs
python scripts/prepare_course_data.py '/path/Computer Vision.zip'
python -m pytest -q
python scripts/execute_notebooks.py
```

Open the notebooks with JupyterLab from this repository. They find the repository root and use the package in `src`. The default budget is short and runs on CPU. The committed results were produced with `PROJECT_FULL=1`. Set `PROJECT_FULL=1` in the environment for longer training. The optional ImageNet transfer route requires `PROJECT_IMAGENET=1`; it downloads official weights and is not part of the default measured run.

## Recorded runs

The notebook outputs, [figures](figures) and [result files](results) come from executed full runs. [Execution details](docs/validation.md) record the environment, checks and limits. Generator sample grids are visual inspections; they are not presented as benchmark-quality generation results. New experiments must be executed again to produce their measurements.

Training and validation use separate subsets. Validation selects checkpoints and thresholds. Final test metrics are recorded after those choices. In the ECG projects, available data support a beat-level split; unseen-patient generalization is not measured.

## Code and sources

Models and reusable training code live in `src`. Tests cover tensor shapes, numerical invariants and saved-model agreement. `python -m ruff check src tests scripts` checks unused names and common errors; formatting uses four-space indentation. Comments explain decisions where the code alone would leave an ambiguity.

The project sequence draws on UniPd Machine Learning for Human Data, UniPd Computer Vision and Cognitive Systems and HSE deep learning exercises. [Source mapping](docs/sources.md) connects each project to the specific course topic. These are independent implementations. Original course solutions, datasets and saved teaching-model weights are not redistributed here. Data access is documented in [datasets.md](docs/datasets.md).
