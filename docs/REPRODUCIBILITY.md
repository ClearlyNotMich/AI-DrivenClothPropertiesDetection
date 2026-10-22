# Reproducibility notes

## Two different workflows

The lightweight tools in the README are a local fabric-metadata demo and a standalone image-tiling helper. Their dependencies are listed in `requirements-tools.txt`. They are separate from the historical training workflow.

The research notebooks are retained as experiment records. Training has not been reproduced during cleanup, and the old Python integration is not a complete application. Do not interpret the lightweight tests as validation of a CNN or of a reported accuracy.

## Historical environment

Notebook metadata records Python 3.10.5 for `Fashion.ipynb`, `Beeld_splitsen.ipynb` and `beeld_resize.ipynb`, and Python 3.9.6 for `Beeld_Voorspelling.ipynb`. There is no recorded package lockfile that establishes the TensorFlow/Keras versions used for the saved runs.

Imports indicate these dependencies:

| Area | Dependencies observed |
| --- | --- |
| Training/notebooks | TensorFlow, Keras, NumPy, pandas, Matplotlib, OpenCV; Jupyter/IPython environment |
| Messaging prototype | `websockets`, `paho-mqtt`, plus research dependencies |
| Local metadata demo | Flask |
| Image preparation helper | OpenCV and its NumPy dependency |

This is an inventory, not a tested training installation command. Establish a compatible environment, then record exact Python/package versions and hardware before reproducing the experiments. Do not assume current Keras and historical APIs are interchangeable.

## Data and paths

1. Obtain a permitted copy of the original photograph dataset. It is not present in Git.
2. Inspect and replace the Windows/Linux workstation paths in the notebooks. Resolve `P2_Fabric_data_csv.csv` relative to the notebook location (`code/`).
3. Establish which specimen IDs and image files belong to each class. The committed CSV has ten records; historical training uses filtered subsets and some model outputs are hard-coded to eight classes.
4. Split photographs or specimens into training, validation and test groups **before** expanding them into patches. Keep augmentations from the same original in the same group.
5. Build class folders explicitly. `Data.sq_cut_img()` groups output by source filename; it does not implement the notebook's CSV-based class grouping or generate `data_processed` for training.

The notebook cells use `os.remove`, `shutil.move`, `os.rmdir` and other filesystem mutations. Inspect each cell before running it and use disposable copies of any images. Fixing `code/images.py` does not remove destructive code from the notebooks.

## Saved results are not a verified benchmark

- The earlier README's 91% claim has no identified matching held-out evaluation run.
- `Fashion.ipynb` contains a saved evaluation of a loaded CNN near 96.50% accuracy. Its execution counts, dataset-loading output and batch counts are inconsistent with a single clean, sequential run.
- `Beeld_Voorspelling.ipynb` contains a saved evaluation near 99.77%, but the evaluated dataset is also used as validation during training. This is not an untouched test set.
- The source-image grouping of patches is not documented. Randomly splitting related patches can inflate performance.

These numbers describe saved output, not newly reproduced results, and should not replace the old headline claim. A future benchmark needs a fixed data manifest, documented split and seed, class mapping, baseline, exact model/preprocessing configuration, dependency versions, and per-class results on an untouched holdout.

## Integration limitations

- `code/main.py` has incomplete model setup/load paths, hard-coded class counts, blank MQTT configuration, and evaluation on the training data. It should not be used as a validated inference service.
- The non-destructive helper does not implement the missing class-collection/training pipeline used by `main.py`.
- Trained `.h5` models are referenced by notebooks and ignored by Git, but are not supplied.
- `code/resnet.py` is an empty placeholder; the actual ResNet experiment is elsewhere.
- The original `interface/` files are historical artifacts. Use the corrected Flask demo in `code/` for the supported metadata view.
- The CSV is read using the historical Latin-1 setting. One textile-description character needs source-encoding confirmation; quoted comma-containing compositions must be parsed with a CSV parser, not string splitting.

## Validation boundary

The lightweight tools were checked on 22 October 2026 with Python 3.12.14, Flask 3.1.3, OpenCV 5.0.0.93 and NumPy 2.5.4. The 16 synthetic-image and Flask regression checks passed in a clean virtual environment installed from `requirements-tools.txt`. This environment is not claimed to reproduce the historical training runs.

The cleanup checks cover safe tiling behavior, metadata parsing and local HTTP responses, Python syntax, and notebook JSON integrity. They do not run training, connect to MQTT/WebSocket services, download model weights, or establish model quality. Original research files and Git history remain available for a later reproducibility pass.
