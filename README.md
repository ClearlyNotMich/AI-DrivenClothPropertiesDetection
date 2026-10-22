# AI-Driven Cloth Properties Detection

A collaborative computer-vision research prototype exploring fabric identification from close-up textile photographs. The repository contains image-preparation experiments, CNN and transfer-learning notebooks, a table of fabric measurements, and an interface prototype.

The team captured the fabric photographs themselves over several weeks, using a camera in a dark room and working in two-hour shifts. The full photograph dataset and trained model weights are **not included** in this repository.

## What is implemented

| Component | Status |
| --- | --- |
| CNN, LeNet-style, VGG19 and ResNet50 experiments | Historical research notebooks and Python prototypes; training has not been reproduced in the cleanup environment. |
| Image preparation | A configurable, non-destructive tiling helper in `code/images.py`, with synthetic-image tests. |
| Fabric metadata demo | A local Flask interface that looks up a fabric identifier in the measurement table. **It does not run a classifier.** |
| MQTT/WebSocket integration | An incomplete prototype in `code/main.py`; not a supported end-to-end application. |

The classification experiments predict fabric identifiers. The measurement table supplies associated composition and physical properties; this snapshot does not establish a model that directly predicts all 44 measurement fields from a new image.

## Contributions and data

The source comments in [`code/main.py`](code/main.py) credit the custom CNN to **Agus**, the LeNet-style model to **Vinesh**, VGG19 transfer learning to **Michal**, and ResNet to **Tracy**. These are the recorded implementation credits, not a complete account of each person's project work. See [provenance and attribution](docs/PROVENANCE.md).

[`code/P2_Fabric_data_csv.csv`](code/P2_Fabric_data_csv.csv) contains ten fabric records and 44 fields, including identifiers, composition, weight, thickness, and bending/stretch/shear measurements. Notebook outputs include sample fabric images and plots. The maintainer has confirmed permission for continued public portfolio publication of the collaborative code, committed data excerpts and their Git history.

## Run the lightweight tools

These tools do not require TensorFlow, model weights or the original dataset. Use Python 3.10 or newer in a virtual environment:

```bash
git clone https://github.com/ClearlyNotMich/AI-DrivenClothPropertiesDetection.git
cd AI-DrivenClothPropertiesDetection
python -m venv .venv
# Linux/macOS:
source .venv/bin/activate
# Windows PowerShell instead: .venv\Scripts\Activate.ps1
python -m pip install -r requirements-tools.txt
```

### Fabric metadata demo

```bash
python code/app.py
```

Open `http://127.0.0.1:5000`. Select a fabric identifier or preview an image whose filename contains an identifier from the CSV, such as `21F00012`. The displayed properties come from the CSV lookup, **not image inference**. Selected images are previewed locally in the browser and are not uploaded by this demo.

### Non-destructive image preparation

```python
import sys
sys.path.insert(0, "code")
from images import Data

preparation = Data(data="data", temp_data="data_temp")
result = preparation.sq_cut_img()
print(result.written)
print(result.skipped)
```

Place a small set of image copies in `data/` before trying this example. By default the helper resizes each image to 5000 × 5000 pixels and produces a 25 × 25 grid of patches. Source files are retained; overlapping directories and existing per-image output paths are rejected. The output is grouped by source image, **not** automatically arranged into fabric classes or train/validation/test splits.

Keep every patch from the same source photograph—and, where appropriate, the same physical fabric specimen—in the same evaluation split. Creating patches first and splitting them independently can leak closely related content between training and evaluation.

### Checks

```bash
python -m unittest discover -s tests -v
```

These tests exercise the lightweight helper and metadata demo. They do not validate model accuracy.

## Explore the research

| File | Purpose |
| --- | --- |
| [`code/Fashion.ipynb`](code/Fashion.ipynb) | Image preparation, fabric-ID classification experiments, model saving and saved evaluations. |
| [`code/Beeld_Voorspelling.ipynb`](code/Beeld_Voorspelling.ipynb) | Earlier CNN classification experiment. |
| [`code/Beeld_splitsen.ipynb`](code/Beeld_splitsen.ipynb) | Historical splitting and regrouping workflow. |
| [`code/beeld_resize.ipynb`](code/beeld_resize.ipynb) | Historical resizing/rotation exploration. |
| [`code/main.py`](code/main.py) | Experimental model and messaging integration. |
| [`interface/`](interface/) | Original standalone interface and deployment snippet, retained as project artifacts. |

Read [reproducibility notes](docs/REPRODUCIBILITY.md) before executing the research notebooks. They contain machine-specific paths and destructive file operations. The corrected helper does not change those historical notebook cells. There is no verified training dependency lockfile, complete dataset, or bundled trained model.

## Results and limitations

The earlier README reported **91% accuracy**, but did not identify the corresponding model, dataset split or evaluation run. That figure is not presented here as a verified benchmark. Saved notebook outputs describe historical experiments; their mixed execution state and uncertain source-image grouping prevent a reproducible generalization claim.

Before publishing a benchmark, record the dataset version and class mapping, group-based split, random seed, dependency versions, model configuration, and held-out evaluation results. Compare against a baseline and report per-class performance and limitations.

## License status

The earlier README stated that the project was MIT licensed, but this repository does not contain the full license text or a record of its copyright holders and coverage. That needs clarification with the contributors. This cleanup does not introduce new license terms. Permission to publish and transfer this portfolio repository is separate from granting a reuse license for its code or data.
