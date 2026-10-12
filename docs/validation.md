# Execution and validation

All 13 project notebooks were executed with `PROJECT_FULL=1` on CPU using seed 1031. The standard supervised budget was 40 epochs. Each notebook states its settings in the first output; filtering and feature-matching experiments do not train a neural network. Total recorded execution time for this repository was 230.7 seconds; hardware contention makes timing descriptive.

The runtime blocked local Jupyter kernel sockets. Execution therefore used the included plain-Python engine with Matplotlib's Agg backend. It ran the same cells in order and captured their stdout and figures into the notebook. Use `PROJECT_FULL=1 python scripts/execute_notebooks.py --inprocess` to repeat this route. Normal Jupyter execution is available by omitting `--inprocess`.

`python -m ruff check src tests scripts` passed, and all 8 unit tests passed after code formatting. The notebooks also check numerical agreement, tensor shapes and their topic-specific invariants. Every result JSON was produced by executed code with non-finite values rejected. The figures were visually inspected, including segmentation overlays and image reconstructions.

The optional ImageNet download experiment was not executed. Its frozen ResNet wrapper was tested with a random backbone to verify tensor flow, gradients and unchanged BatchNorm statistics. That structural test does not claim pretrained performance.

Package versions are in `docs/environment.json` and `requirements-tested.txt`. `results/execution_report.json` records each executed notebook. Metrics measure the datasets and splits described in the notebooks. Synthetic segmentation scores do not establish performance on real microscopy images.
