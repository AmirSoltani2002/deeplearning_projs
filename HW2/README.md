# HW2 - CNN Classification, Augmentation, and Transfer Learning

Two image-classification studies examine how CNN architecture, initialization, regularization, and data augmentation influence performance.

**Sources:** [Assignment](NNDL_HW2.pdf) · [Submitted report](report.pdf)

## Question 1 - Brain MRI classification

The assignment asks for three CNN architectures from a reference paper and comparisons of data splits, dropout, and Glorot initialization. Although the assignment describes Alzheimer's detection broadly, the submitted data and code classify **AD versus MCI**, rather than AD versus a healthy control group.

### Approach

- Load **1,654 MRI images**: 965 AD and 689 MCI images, as counted in the report.
- Resize images to **64 × 64**, use one grayscale channel, and scale pixels to `[0, 1]`.
- Generate four additional variants per image, expanding the data fivefold.
- Implement the paper's **Testing Model 1**, **Testing Model 2**, and **Proposed Model**, plus dropout and alternative-initialization variants.
- Train with Adam, learning rate `0.001`, categorical cross-entropy, and a two-class softmax output. The notebook uses 20 epochs and batch size 64 for these comparisons.
- Analyze learning curves, confusion matrices, classification reports, and ROC/AUC.

### Reported findings

| Comparison | Reported ROC AUC |
| --- | --- |
| Proposed model, 30% held out | Approximately 0.81 |
| Proposed model, 50% held out | Approximately 0.79 |
| Proposed model with dropout | Approximately 0.83 |
| Proposed model without Glorot initialization | Approximately 0.80 |

The report describes overfitting in the baseline and an improvement after adding dropout. It also reports a baseline F1 of approximately 0.82, rising to approximately 0.83 with dropout.

**Evaluation scope:** augmentation precedes image-level splitting in the notebook, so related image variants may appear in different partitions. No subject-disjoint split is established. These coursework results describe this experiment and should not be treated as evidence of patient-level clinical performance.

## Question 2 - Augmentation with pretrained CNNs

Compare **VGG16** and **ResNet50** on a small cats-versus-dogs dataset, with and without augmentation.

The report describes 350 cat and 350 dog images in total, including a fixed test set of 100 images per class. The remaining 500 images are split 70/30 into training and validation data. Images are resized to **128 × 128 RGB**.

The solution replaces each pretrained classifier with a small dense classification head, first trains with the backbone frozen, and then fine-tunes its final convolutional block. Augmentation adds horizontal flips, zoom, and rotation. The notebook trains for 25 epochs in each phase, using batches of 10.

### Reported accuracy

From the report's Question 2 results table, printed page 18:

| Model | Augmentation | Training | Validation | Test |
| --- | --- | --- | --- | --- |
| VGG16 | Yes | 95% | 92% | 94% |
| VGG16 | No | 99% | 92% | 95% |
| ResNet50 | Yes | 89% | 86% | 83% |
| ResNet50 | No | 93% | 83% | 75% |

Augmentation improves ResNet50's test accuracy in these runs, while VGG16's test accuracy changes from 95% to 94%. The results therefore support a model-dependent effect rather than a uniform improvement from augmentation.

## Files

| File | Contents |
| --- | --- |
| [1.ipynb](1.ipynb) | MRI preprocessing, augmentation, CNN variants, and evaluation |
| [2.ipynb](2.ipynb) | VGG16/ResNet50 transfer learning and augmentation comparisons |
| [Q1 Dataset/](Q1%20Dataset/) | Included AD/MCI MRI image folders |

## Running

Use `HW2/` as the notebook working directory. Libraries imported by the notebooks include **NumPy, pandas, Matplotlib, TensorFlow/Keras, KerasCV, OpenCV, scikit-learn, and PyTorch**.

The MRI notebook reads the included `Q1 Dataset/Alzheimer_s_Disease_Neuroimaging_ADNI_Dataset/AD` and `MCI` directories.

The cats-versus-dogs notebook needs these external inputs, which are absent from the repository:

```text
HW2/
├── Data/
│   ├── Train/Cats/
│   ├── Train/Dogs/
│   ├── Test/Cats/
│   └── Test/Dogs/
├── resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5
└── vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5
```

Provide the local weight files expected by the notebook, or intentionally change the loading cells to download the corresponding pretrained weights. The notebooks use older Keras APIs, and no fresh training run or dependency compatibility check was performed for this documentation update.
