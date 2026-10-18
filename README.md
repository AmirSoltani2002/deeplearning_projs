# Neural Networks and Deep Learning Projects

Course projects from **Spring 2024**, covering neural-network fundamentals, computer vision, natural language processing, time-series modeling, and generative learning.

The reports identify **Amirabbas Reza Soltani** and **Nima Niroumand** as the authors. Each project README explains the assignment, the submitted approach, and the available results in English. The original assignment sheets and reports are in Persian.

## Projects

| Project | Main topics | Experiments |
| --- | --- | --- |
| [HW1](HW1/README.md) | Neural-network fundamentals and robustness | McCulloch-Pitts logic, FGSM/PGD attacks, Adaline/Madaline, regression and classification capacity experiments |
| [HW2](HW2/README.md) | CNNs and data augmentation | AD/MCI brain MRI classification; VGG16 and ResNet50 transfer learning for cats versus dogs |
| [HW3](HW3/README.md) | Segmentation and object detection | Brain MRI U-Net; a custom Faster R-CNN-style underwater detector |
| [HW4](HW4/README.md) | Sequence learning | Persian tweet emotion recognition; engine health classification and remaining useful life estimation |
| [HW5](HW5/README.md) | Transformer transfer learning | BERT/CT-BERT fake-news classification; VGG19 versus DeiT on CIFAR-10 |
| [HW6](HW6/README.md) | Generative models | VAE, conditional VAE, VQ-VAE, and Pix2Pix image translation |
| [HWE](HWE/README.md) | Additional experiments | Autoencoder clustering, FaBERT augmentation, wake-word detection, U-Net versus TA-Unet |

## Reading the projects

For each folder, start with its assignment PDF, then read `report.pdf`, and use the notebook links in the project README to inspect the implementation. **HW3 has no report in this repository**; its README distinguishes the assignment requirements from the notebook implementation.

Results labeled **reported** are transcribed from the submitted reports, rather than measurements from a new run. Some reports and saved notebook outputs differ; relevant differences are called out in the project READMEs. These are coursework experiments, and their results depend on the original data splits, preprocessing, and runtime.

## Running the notebooks

1. Clone the repository and open the project you want to explore:

   ```bash
   git clone https://github.com/AmirSoltani2002/deeplearning_projs.git
   cd deeplearning_projs
   jupyter lab
   ```

2. Use a Python environment with Jupyter and the libraries listed in that project's README. The notebooks use a mixture of TensorFlow/Keras and PyTorch; there is no repository-wide dependency lockfile or verified environment specification.
3. Obtain any external datasets and pretrained weights, then update the notebook's data paths. Some notebooks expect Google Colab, Google Drive, Kaggle, or files from the original author's machine.
4. Set the working directory to the notebook's folder and inspect its setup cells before executing. A GPU is useful for the larger image and transformer experiments; some cells explicitly select CUDA.

MNIST, Fashion-MNIST, and CIFAR-10 are loaded through dataset APIs. The Wine CSV and HW2 MRI images are included; most other datasets, downloaded weights, and audio recordings are external. See each project README for specifics.

The documentation was checked against the PDFs, notebook source, and selected saved outputs. Full training and fresh metric reproduction were not performed as part of the README update.
