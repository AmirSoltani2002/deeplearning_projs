# HW1 - Neural-Network Fundamentals and Adversarial Robustness

Implement early neural-network models, examine adversarial attacks on handwritten-digit recognition, and study how training-data size and network depth affect learning.

**Sources:** [Assignment](NNDL-HW1.pdf) · [Submitted report](report.pdf)

## Questions and approach

### 1. Two's complement with McCulloch-Pitts neurons

The assignment asks for a network that computes the two's complement of a four-bit binary input. The solution defines a threshold-neuron class, builds the required logical operations, and evaluates the network on all 16 input combinations. The report explains the connections and weights.

### 2. Adversarial attacks on MNIST

Train a fully connected digit classifier and investigate white-box, untargeted attacks:

- Normalize the 28 × 28 grayscale images and inspect the class distribution.
- Use a dense network with hidden widths 512, 128, and 32, followed by a ten-class output.
- Generate **FGSM** perturbations with `epsilon = 0.1`.
- Generate **PGD** perturbations with `epsilon = 0.1`, `alpha = 0.005`, and 20 iterations.
- Compare predictions and visualize original and perturbed images.

The report gives approximately **91%** accuracy before attack, **10%** after FGSM, and **6%** after PGD. The notebook generates and evaluates the perturbations on `X_train`; these figures should not be interpreted as an independent adversarial test-set benchmark. The original MNIST test set is evaluated separately in the notebook.

### 3. Adaline and Madaline

Implement **Adaline** for two binary versions of the Wine problem: class 1 versus the rest, and class 2 versus the rest. The solution uses the Alcohol and Malic acid features, standardizes the inputs, and plots learning curves and decision boundaries.

The **Madaline** experiment uses synthetic, nonlinearly separable Gaussian-quantile data and the **MRI learning rule**. Networks with 3, 5, and 8 hidden neurons illustrate how multiple linear units can represent a more complex boundary. The report also discusses the difference between MRI and MRII.

Reported Adaline accuracies are approximately **85%** for class 1 versus the rest and **89%** for class 2 versus the rest. These are the report's evaluation figures, not newly reproduced scores.

### 4. Training-data size, depth, and overfitting

- **Regression:** approximate a sine curve, vary the train/test proportions, increase network depth, and search over layer counts with SciKeras and `GridSearchCV`.
- **Classification:** vary the amount of MNIST training data and the number of dense layers, then compare loss and accuracy curves.

The report selects **13 hidden layers** in its regression grid-search experiment and discusses reduced generalization when model capacity and available training data are poorly matched. The folder also includes two videos visualizing the regression experiments.

## Files

| File | Contents |
| --- | --- |
| [1.ipynb](1.ipynb) | McCulloch-Pitts neurons and four-bit two's complement |
| [2.ipynb](2.ipynb) | MNIST classifier, FGSM, and PGD |
| [3.ipynb](3.ipynb) | Wine Adaline and synthetic-data Madaline |
| [4.1.ipynb](4.1.ipynb) | Sine regression, capacity experiments, and grid search |
| [4.2.ipynb](4.2.ipynb) | MNIST data-size and depth experiments |
| [wine.data.csv](wine.data.csv) | Wine data used by `3.ipynb` |
| [change_proportion.gif.mp4](change_proportion.gif.mp4) | Regression visualization for changing data proportions |
| [Change_layers.gif.mp4](Change_layers.gif.mp4) | Regression visualization for changing depth |

## Running

Open the notebooks with `HW1/` as the working directory. The main libraries are **NumPy, pandas, Matplotlib, scikit-learn, TensorFlow/Keras, SciKeras, and Pillow**. MNIST is downloaded through Keras; the Wine CSV is already included.

The regression notebook writes plots into `proportion/` and `layer/`; create these directories before running the corresponding cells. Its checkpoint filenames and SciKeras integration may require adjustment for the installed Keras/SciKeras versions. No dependency versions are pinned, and a fresh end-to-end run has not been verified for this documentation update.
