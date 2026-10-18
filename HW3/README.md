# HW3 - Brain MRI Segmentation and Underwater Object Detection

Explore pixel-level segmentation with U-Net and bounding-box prediction with a custom two-stage detector.

**Source:** [Assignment](HW3.pdf). **No `report.pdf` is present in this folder.** The implementation descriptions below come from the notebooks; assignment requirements are identified separately, and no report-based performance scores are available.

## Question 1 - Brain MRI segmentation

### Assignment

Segment brain MRI images using U-Net. The question requests an 80/10/10 train/validation/test split, suitable augmentation, Dice and IoU metrics, learning curves, and comparisons of predicted masks with the reference masks.

### Notebook implementation

[1.ipynb](1.ipynb) provides a dataset class and a wrapper for loading image/mask pairs, splitting data, training, testing, saving the model, and visualizing outputs.

- Images are resized to **128 × 128**.
- The model is `segmentation_models_pytorch.Unet` with an **ImageNet-pretrained ResNet34 encoder**. The encoder parameters are frozen.
- Training uses **binary cross-entropy with logits**, **Adam**, and learning rate `1e-4`.
- The wrapper implements Dice, IoU, and pixel-accuracy calculations and records training and validation histories.

This implementation uses a library U-Net with a pretrained encoder; it should not be described as a fully handwritten reproduction of the original U-Net architecture.

## Question 2 - Underwater object detection

### Assignment

Build a Faster R-CNN-style detector for an underwater dataset described as **638 images across seven classes**. Required work includes exploratory analysis, augmentation, anchor construction, positive/negative anchor assignment, a pretrained ResNet101 backbone, GIoU-related comparisons, model training, and visual evaluation of predicted boxes. Mosaic augmentation is an optional extension.

### Notebook implementation

[2.ipynb](2.ipynb) contains a custom PyTorch pipeline with:

- An underwater dataset loader that reads images and YOLO-style text annotations from `train`, `valid`, and `test` folders.
- Conversion of normalized center/width/height annotations into corner-coordinate boxes.
- A pretrained **ResNet101** feature extractor.
- Anchor-point and anchor-box generation, overlap calculations, and positive/negative anchor selection.
- Proposal, classification, and box-regression modules, plus a two-stage detector wrapper.
- Training and inference/evaluation code.

The notebook contains implementation work toward the requested detector, but its presence alone does not establish that every assignment component is complete or that training runs successfully.

## Files

| File | Contents |
| --- | --- |
| [HW3.pdf](HW3.pdf) | Original questions and dataset/reference links |
| [1.ipynb](1.ipynb) | MRI segmentation pipeline |
| [2.ipynb](2.ipynb) | Underwater detection pipeline |

## Running and known issues

The main imports are **PyTorch, torchvision, segmentation-models-pytorch, Albumentations, NumPy, pandas, Matplotlib, Pillow, OpenCV, scikit-learn, scikit-image, and tqdm**.

1. Obtain the MRI image/mask dataset referenced by the assignment and update the local `kaggle_3m` path in `1.ipynb`. Its loader renames source image files with `os.rename`; use a working copy of the data. Create the `pics/` output directory before plotting.
2. Correct the import typo `train_test_splito` to `train_test_split` before running the segmentation notebook. Albumentations imports may also need adaptation to the installed version.
3. For detection, mount Google Drive in Colab or replace the mount and `/content/drive/MyDrive/aquarium_pretrain` paths. Each dataset split must contain `images/` and matching `labels/` files.
4. Review the detector's intermediate data-handling cells before a full run. A saved output contains a `KeyError: 1`, and the custom detector has not been verified end to end in this documentation update.

Both datasets are external to this repository. Quantitative results are intentionally omitted because no submitted report is available and no fresh evaluation was run.
