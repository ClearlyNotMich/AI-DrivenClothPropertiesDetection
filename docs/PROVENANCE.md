# Provenance and attribution

## Collaborative work

The existing comments in `code/main.py` record these model implementation credits:

| Contributor | Recorded credit |
| --- | --- |
| Agus | Custom CNN (`model_one`) |
| Vinesh | LeNet-style model (`model_two`) |
| Michal | VGG19 transfer learning (`model_three`) |
| Tracy | ResNet (`model_four`) |

The repository's import commits do not capture the complete collaborative development process. The table preserves the source's first-name credits; it does not imply that these were each person's only contributions or that every component was written by one person.

## Fabric data

The maintainer described the photographs as team-collected: the team used a camera in a dark room over several weeks, working in two-hour shifts. The original full-resolution dataset is not committed. Notebook paths such as `vizoo_images/22V_5XP10_T15_filtered` describe the historical local workflow; they do not by themselves establish a third-party owner or license.

The repository includes:

- `code/P2_Fabric_data_csv.csv`: ten fabric records, 44 fields, including measurements and material descriptions.
- Sample fabric imagery and evaluation plots embedded in notebook outputs.
- Interface placeholder images.

On 22 October 2026, the maintainer confirmed authority or contributor permission to keep the collaborative project public and transfer the repository between their GitHub accounts, retaining contributor credit. They also confirmed that applicable course/employer/client/sponsor rules permit this, or do not apply, and that publication permission covers the committed measurement table, embedded notebook images/outputs, and retained Git history.

The photograph collection account does not specify the measurement equipment, measurement protocol, or exact creation process for the CSV. Document those details before presenting a fully reproducible dataset. Publication permission does not establish unrestricted downstream reuse rights or authorize uploading the full dataset.

## Dependencies and licensing

The research imports TensorFlow/Keras architectures, including ImageNet-initialized VGG19 and ResNet50. Those external packages and pretrained weights have their own terms. They are not bundled as trained weights in this repository. The original interface references Bootstrap and jQuery CDNs.

The earlier README stated MIT licensing, but no complete LICENSE file or documented copyright coverage accompanies it. The intended code license and the terms for dataset reuse remain to be clarified. No new license is applied by this cleanup; existing commit history is retained.
