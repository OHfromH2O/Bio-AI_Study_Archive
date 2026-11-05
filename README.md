# Bio-AI Study Archive

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow%20%7C%20Keras-FF6F00?logo=tensorflow&logoColor=white)
![Focus](https://img.shields.io/badge/Focus-Bio--AI%20%7C%20Medical%20Imaging-green)

I trained as a wet-lab biologist and I am moving toward computational work. This
repository holds the code I wrote while teaching myself deep learning. Everything
here is something I actually ran, so the notebooks include the versions that did
not work as well as the ones that did.

## What is in this repository

| Folder | Contents | Files |
|---|---|---|
| [`PyTorch_Pracs/`](PyTorch_Pracs) | Six end-to-end projects. Each one is kept as a versioned record of baseline, diagnosis, and fix | 20 notebooks |
| [`TensorFlow_Pracs/`](TensorFlow_Pracs) | Chapter exercises from *Everyone's Deep Learning*, covering perceptrons through transformers | 23 notebooks |
| [`Vibe coding_Pracs/`](Vibe%20coding_Pracs) | Three small applications built with an AI coding assistant, each with its own spec | 3 projects |

## Results worth looking at

These are the measured numbers from the project folders, not targets.

| Project | Task | Result |
|---|---|---|
| [COVID-19 CT classification](PyTorch_Pracs/6_LAIDD-Medical-Imaging-CNN) | Chest CT, COVID vs non-COVID. Custom CNN, then a residual CNN, then ResNet50 feature fusion | ROC AUC 0.887, accuracy 80.3%, sensitivity 81.9% |
| [CIFAR-10 transfer learning](PyTorch_Pracs/5_CNN_transfer_learning_model) | Hand-built CNN against ResNet50 fine-tuning, two revisions of each | 89.02% custom (+17.47 pp), 95.27% transfer (+10.47 pp) |
| [Breast cancer diagnosis](PyTorch_Pracs/2_Breast_Cancer_Binary_Classification_model) | Binary classification with K-fold cross-validation and Optuna tuning | AUC 0.9954, malignant recall 0.976 |
| [Insurance cost prediction](PyTorch_Pracs/4_Insurance_Cost_Prediction_model) | Tabular regression, 5-fold cross-validation | R² 0.8574 ± 0.0298 |

The medical imaging project is the one closest to what I want to do next. It uses
the COVID-CT dataset and builds up through three pipelines, combining 2048
features from a pretrained ResNet50 with 128 features from a custom extractor.

## How these projects are written

Each project folder keeps the earlier versions instead of only the final one,
because the useful part of the exercise was finding out why a version failed.
Two examples:

The diabetes regression model reached an R² of 0.4501, which means it explained
less than half the variance. I recorded that as insufficient rather than
presenting it as a result.

The wine classifier reached 97% accuracy, but the test set held only about 36
samples, so a single misclassification moves the number by 2.8 percentage
points. The folder says so, and treats the accuracy as too uncertain to claim.

## Coursework behind this

The table separates what produced code in this repository from what did not, so
that nobody goes looking for files that are not here.

| Course or book | Provider | Period | Code in this repository |
|---|---|---|---|
| Deep Learning Specialization | DeepLearning.AI (Coursera) | Oct – Dec 2024 | No |
| Everyone's Deep Learning | Taeho Jo ([source](https://github.com/taehojo/deeplearning)) | — | `TensorFlow_Pracs/` |
| Medical imaging for diagnosis and biomarker discovery | LAIDD | — | `PyTorch_Pracs/6_LAIDD-Medical-Imaging-CNN/` |
| Big Data Analysis and AI Development (NCS) | Winspec | Apr – May 2026 | No |

## A note on language

The README files inside each project folder are written in Korean, because I
wrote them while studying. This top-level file is in English. If you want the
detail behind any number in the table above, the project folder has it.

## Related repositories

Applied work that grew out of this archive:

- [vhh-mpnn-selfconsistency](https://github.com/OHfromH2O/vhh-mpnn-selfconsistency) — whether ProteinMPNN redesigns fold back to the original structure
- [pdl1-rfdiffusion-hotspots](https://github.com/OHfromH2O/pdl1-rfdiffusion-hotspots) — whether RFdiffusion hotspot specification changes binder design outcomes
