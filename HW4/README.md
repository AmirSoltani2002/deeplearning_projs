# HW4 - Persian Emotion Recognition and Predictive Maintenance

Study sequence models in two settings: recognizing emotions in Persian tweets and estimating the health and remaining useful life of turbofan engines.

**Sources:** [Assignment](NNDL_HW4.pdf) · [Submitted report](report.pdf)

## Question 1 - Persian tweet emotion recognition

### Task and data

Classify Persian tweets into six emotions: **anger, disgust, fear, joy, sadness, and surprise**. The report describes **113,829 tweets** with an imbalanced class distribution.

The assignment requests text preprocessing, ParsBERT-based representations, a CNN-LSTM hyperparameter search, comparisons with standalone CNN and LSTM models, and optional classical machine-learning baselines.

### Approach

- Remove URLs and HTML, reduce repeated characters, convert emojis to text, normalize Persian text, remove stopwords, and apply stemming.
- Use the `HooshvareLab/bert-fa-zwnj-base` tokenizer and model, sequence length 32, and a linear projection from 768 to **120 features per token**.
- Compare **CNN-LSTM**, **CNN**, and **LSTM** classifiers.
- Search eight combinations of batch size `{8, 64}`, learning rate `{0.001, 0.0001}`, and optimizer `{Adam, SGD}`. The report selects Adam, batch size 8, and learning rate `0.001`.
- Add TF-IDF baselines with Naive Bayes, a linear SVM, and logistic regression. The report says these classical experiments use **10% of the data** because of resource limits.

### Reported results

The report labels the averaged classification metrics as macro averages. Accuracy and F1 are reproduced below from its Question 1 tables, printed pages 9-10:

| Model | Accuracy | Reported F1 |
| --- | --- | --- |
| CNN-LSTM | 0.69 | 0.68 |
| CNN | 0.62 | 0.60 |
| LSTM | 0.50 | 0.38 |
| TF-IDF + Naive Bayes | 0.50 | 0.46 |
| TF-IDF + linear SVM | 0.92 | 0.92 |
| TF-IDF + logistic regression | 0.89 | 0.87 |

The hybrid model leads the neural models in the report. The classical results use a smaller sampled dataset and should not be read as a controlled comparison on identical evaluation data. Saved notebook outputs also differ from some report figures; this table preserves the submitted report's values.

## Question 2 - Turbofan health and remaining useful life

### Task and preprocessing

Use **NASA C-MAPSS FD001** engine time series for two tasks:

- **Classification:** unhealthy when remaining useful life (RUL) is below 50 cycles, otherwise healthy.
- **Regression:** predict RUL in cycles.

The report describes 20,631 training rows and 13,096 test rows. The solution removes low-variation channels, keeps **18 input features**, applies min-max scaling, and constructs **30-step windows**. Training RUL is derived from each engine's last cycle; test RUL additionally uses the supplied `RUL_FD001` offsets.

### Models and evaluation

Compare a **CNN-LSTM hybrid** with CNN-only and LSTM-only baselines. The submitted implementation trains for up to **200 epochs**, with batch size **200**, and compares runs with and without early stopping. Classification uses Adam and binary cross-entropy; regression uses RMSprop and mean squared error.

Classification is assessed with per-class precision, recall, F1, confusion matrices, and ROC curves. Regression is evaluated on both **all test windows** and the **last window per engine**.

### Selected reported results

| Classification model | Overall accuracy | Unhealthy-class F1 |
| --- | --- | --- |
| CNN-LSTM without early stopping | 0.96 | 0.80 |
| CNN-LSTM with early stopping | 0.96 | 0.76 |
| CNN with early stopping | 0.96 | 0.79 |
| LSTM with early stopping | 0.95 | 0.64 |

| Regression model | All-window RMSE | All-window MAE |
| --- | --- | --- |
| CNN-LSTM without early stopping | 46.78 | 34.40 |
| CNN-LSTM with early stopping | 39.94 | 29.93 |
| CNN with early stopping | 45.13 | 32.11 |
| LSTM with early stopping | 43.07 | 32.14 |

Errors are in cycles. For the hybrid model, last-window RMSE is **25.58 without early stopping** and **26.46 with early stopping**. Early stopping improves the all-window regression score in the report, but does not improve every evaluation view or classification metric.

## Files

| File | Contents |
| --- | --- |
| [1.ipynb](1.ipynb) | Persian preprocessing, representations, neural classifiers, and TF-IDF baselines |
| [2.ipynb](2.ipynb) | FD001 preprocessing, windowing, health classification, and RUL regression |

## Running

For `1.ipynb`, the imports include **PyTorch, Transformers, TensorFlow/Keras, Hazm, NLTK, emoji, Beautiful Soup, scikit-learn, NumPy, pandas, and Matplotlib**. Replace the Google Drive path to `persian-tweets-emotional-dataset` with your dataset location. Its embedding cells call `.cuda()` explicitly; use a CUDA-capable runtime or adapt device handling.

For `2.ipynb`, use **TensorFlow/Keras, NumPy, pandas, Matplotlib, and scikit-learn**. From `HW4/`, provide:

```text
Dataset/
├── train_FD001.csv
├── test_FD001.csv
└── RUL_FD001.csv
```

The CSV column names must match those used in the preprocessing cells; the notebook drops the first CSV column as an index. These datasets are not included. Dependency versions are unpinned, and full training was not rerun for this documentation update.
