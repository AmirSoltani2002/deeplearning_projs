# HWE - Additional Deep-Learning Experiments

Four extra-credit projects explore representation learning, Persian-language augmentation, audio classification, and attention-based segmentation.

**Sources:** [Assignment](HWe.pdf) · [Submitted report](report.pdf)

## Question 1 - Learning image representations for clustering

### Task and approach

Train a **convolutional autoencoder** separately on **MNIST** and **Fashion-MNIST**, then cluster the learned representations to investigate whether visual categories emerge without using class labels for training.

- Normalize 28 × 28 images and reserve 25% of the training data for validation.
- Use a **five-dimensional latent representation**.
- Train with Adam, learning rate `0.001`, batch size **256**, and **10 epochs**.
- Evaluate reconstruction using MSE and MAE and compare originals with reconstructed images.
- Extract latent features, run **K-means for 5-15 clusters**, and choose the cluster count using the **silhouette score**.
- Inspect the relationship between discovered clusters and the known dataset labels.

### Reported results

| Dataset | Test reconstruction MSE | Test reconstruction MAE | Selected clusters |
| --- | --- | --- | --- |
| MNIST | 0.029 | 0.070 | 8 |
| Fashion-MNIST | 0.028 | 0.094 | 8 |

The report observes that visually similar digits or clothing categories can share clusters. The clustering stage combines features from the training, validation, and test partitions; it is an exploratory analysis of that pooled representation, not a held-out classification score.

## Question 2 - Back-translation for FaBERT sentiment analysis

### Task and approach

Classify **DeepSentiPers** Persian product reviews after merging five sentiment labels into **negative, neutral, and positive** classes.

- Inspect class imbalance and token lengths.
- Compare the original data with a version that oversamples the minority negative class.
- Augment text by translating **Persian → English → Persian** with `googletrans`.
- Use `sbunlp/fabert` with a three-class classification head.
- Keep most pretrained layers frozen and train the final encoder layer and classifier for **5 epochs**, using AdamW.
- Compare accuracy, F1, confusion matrices, and misclassified reviews.

### Reported results

| Training-data variant | Validation accuracy | Reported F1 |
| --- | --- | --- |
| Balanced through oversampling | Approximately 73% | Approximately 0.75 |
| Augmented with back-translation | Approximately 81% | Approximately 0.78 |

The report describes poor recognition of the minority class before balancing. Its augmented experiment improves both reported metrics over the balanced experiment. These figures describe the submitted experimental splits; a fresh run with original examples and their translated/duplicated variants grouped before splitting would be needed for an independent generalization estimate.

## Question 3 - Wake-word detection

### Task and approach

Recognize the word **“Awake”** versus other speech or background audio.

- Record **100 two-second wake-word clips** and **100 two-second background/other-sound clips**.
- Remove silence, normalize audio, and extract mel-spectrogram representations.
- Apply spectrogram time shifts, frequency shifts, and noise augmentation.
- Train a **CNN** with a binary output using Adam, binary cross-entropy, **20 epochs**, and batch size **8**.
- Select the checkpoint using validation accuracy.

The report describes **120 training, 40 validation, and 40 test clips**, with **95% test accuracy** and **0.60 test loss**. This is a small recorded dataset; speaker-independent or environment-independent performance is not established by the report.

## Question 4 - U-Net versus TA-Unet for underwater segmentation

### Task and approach

Perform **eight-class semantic segmentation** on **SUIM** underwater images and compare U-Net with **TA-Unet**, which adds **triplet attention**.

- Resize images and masks to **128 × 128** and apply paired augmentation.
- Convert RGB-coded masks to class indices and one-hot targets.
- Reserve 10% of the training data for validation and use the provided test set for final evaluation.
- Train both models with Adam and cross-entropy for **20 epochs**.
- Compare learning curves and mean intersection over union (**mIoU**).

| Model | Reported test mIoU |
| --- | --- |
| U-Net | Approximately 0.56 |
| TA-Unet | Approximately 0.58 |

The attention model has a modest advantage in the submitted experiment. These are the report's values, rather than independently recomputed mIoU scores.

## Files

| Notebook | Contents |
| --- | --- |
| [Q1.ipynb](Q1.ipynb) | Autoencoders, latent features, K-means, and silhouette analysis |
| [FaBert.ipynb](FaBert.ipynb) | Sentiment preprocessing, balancing, translation, and FaBERT training |
| [Q3.ipynb](Q3.ipynb) | Audio recording, feature extraction, augmentation, and CNN classification |
| [TAUnet.ipynb](TAUnet.ipynb) | SUIM preprocessing, U-Net, triplet attention, and evaluation |

## Running

Use the working directory and runtime appropriate to each notebook:

| Notebook | Main libraries | Data and runtime setup |
| --- | --- | --- |
| `Q1.ipynb` | TensorFlow/Keras, NumPy, pandas, Matplotlib, seaborn, scikit-learn | MNIST and Fashion-MNIST download through Keras; silhouette calculations over pooled features can be expensive |
| `FaBert.ipynb` | PyTorch, Transformers, Hazm, googletrans, scikit-learn, pandas, Matplotlib, tqdm; additional TensorFlow/torchvision imports | Replace Google Drive paths under `HWE_2/`; provide `original.csv` and `test.csv`; model downloads and live translation require network access |
| `Q3.ipynb` | TensorFlow/Keras, PyAudio, librosa, NumPy, Matplotlib, scikit-learn | Run locally with a microphone and PortAudio support, or provide existing audio and skip recording cells; the notebook uses `wake_words/` and background-audio inputs |
| `TAUnet.ipynb` | TensorFlow/Keras, Albumentations, OpenCV, NumPy, Matplotlib, seaborn, scikit-learn; additional PyTorch/torchvision imports | Replace Kaggle paths with the SUIM `train_val/images`, `train_val/masks`, `TEST/images`, and `TEST/masks` locations |

Only the built-in MNIST/Fashion-MNIST downloads are self-contained through dataset APIs; review data, recorded audio, and SUIM are external. Older APIs, including the Transformers `AdamW` import, `googletrans` version, and Albumentations imports, may require a compatible environment. Dependencies are unpinned, and full training was not rerun for this documentation update.
