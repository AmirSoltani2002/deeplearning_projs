# HW5 - Transformer Transfer Learning for Text and Images

Compare frozen feature extraction with fine-tuning for fake-news detection, then compare a pretrained convolutional model with a vision transformer on CIFAR-10.

**Sources:** [Assignment](NNDL_HW5.pdf) · [Submitted report](report.pdf)

## Question 1 - COVID-19 fake-news detection

### Task and approach

Classify COVID-related social-media text as real or fake. The assignment asks for three architectures under two transfer-learning strategies:

| Architecture | Representation and classifier |
| --- | --- |
| BERT + dense head | Final `[CLS]` representation followed by a binary classifier |
| BERT + BiGRU | Token representations followed by a bidirectional GRU and binary classifier |
| CT-BERT + BiGRU | COVID-domain pretrained BERT followed by a bidirectional GRU and binary classifier |

The notebook uses `bert-base-uncased` and `digitalepidemiologylab/covid-twitter-bert`. Preprocessing converts emojis to text and tokenizes sequences to a maximum length of **128**.

In **feature-based learning**, the pretrained encoder is frozen and only the added classifier is trained. In **fine-tuning**, the pretrained encoder is updated alongside the classifier. The implementation compares all six combinations and includes learning curves, F1 evaluation, confusion matrices, and example errors.

### Reported results

From the Question 1 result tables, printed pages 1-5:

| Architecture | Feature-based accuracy | Feature-based F1 | Fine-tuned accuracy | Fine-tuned F1 |
| --- | --- | --- | --- | --- |
| BERT + dense head | 0.86 | 0.87 | 0.95 | 0.96 |
| BERT + BiGRU | 0.92 | 0.92 | 0.96 | 0.97 |
| CT-BERT + BiGRU | 0.91 | 0.92 | 0.97 | 0.98 |

Fine-tuning improves all three models in the report, and fine-tuned CT-BERT + BiGRU has the highest reported accuracy. Under frozen feature extraction, BERT + BiGRU has slightly higher reported accuracy than CT-BERT + BiGRU.

The report contains swapped section headings and several mislabeled confusion-matrix captions. This table follows the individual result-table labels and accompanying discussion. Some confusion-matrix counts do not reproduce the adjacent summary accuracy exactly; the values above are the reported summary figures, not recalculated or newly reproduced measurements.

## Question 2 - VGG19 versus DeiT on CIFAR-10

### Task and approach

Investigate transfer learning by updating selected late blocks of pretrained image models. Both notebooks load CIFAR-10 through Keras, resize inputs to **224 × 224**, normalize pixel values, and attach a ten-class classifier.

- **CNN:** ImageNet-pretrained **VGG19**, with its fifth convolutional block trainable and the earlier blocks frozen. The added head uses 256 hidden units, dropout, and softmax classification. Training runs for **10 epochs**.
- **Transformer:** `facebook/deit-base-distilled-patch16-224`, with the first 11 encoder blocks explicitly frozen and the final encoder block trainable, plus a new classification head. Training runs for **3 epochs**.
- Both use Adam with learning rate `0.0001` and batch size **64**. The notebooks record per-epoch training and validation times.

### Reported results

From the report's Question 2 results, printed pages 12-15:

| Model | Training accuracy | Validation accuracy | Training time per epoch | Validation time per epoch |
| --- | --- | --- | --- | --- |
| VGG19 | 97.8% | 85.2% | 298.66 s | 25.21 s |
| DeiT | 98.2% | 96.7% | Approximately 1,347 s | 137 s |

The report favors DeiT's validation accuracy while noting its longer runtime. These timings describe the original runtime and are not hardware-independent benchmarks.

**Comparison details:** the saved CNN notebook holds out **10%** of the CIFAR-10 training set, while the DeiT notebook holds out **20%**. They also use different epoch counts. The scores therefore come from different validation partitions and training budgets. The report's CIFAR-10 sample counts contain apparent transcription errors; the notebooks call the standard dataset loader. The saved final DeiT validation accuracy is approximately **96.88%**, slightly different from the report's **96.7%**.

## Files

| Notebook | Contents |
| --- | --- |
| [code/Q1/code Q1.ipynb](code/Q1/code%20Q1.ipynb) | BERT/CT-BERT models and both transfer-learning strategies |
| [code/Q2/CNN-transferlearning.ipynb](code/Q2/CNN-transferlearning.ipynb) | VGG19 transfer learning |
| [code/Q2/VIT.ipynb](code/Q2/VIT.ipynb) | Distilled DeiT transfer learning |

## Running

The main libraries are **TensorFlow/Keras, Transformers, NumPy, pandas, scikit-learn, Matplotlib, emoji, and Plotly**. The text notebook also imports **PyTorch** and **datatable**. Some model integrations rely on older TensorFlow/Transformers/Keras APIs; no version-compatible environment is pinned.

Run the text notebook from `code/Q1/` with external `Train.csv`, `Test.csv`, and `Val.csv` files in that directory. The loading cells expect a `tweet` column and the label schema used in the notebook.

Run the image notebooks from `code/Q2/`. CIFAR-10 and pretrained model weights are downloaded by the corresponding APIs. A GPU is useful, especially for CT-BERT and DeiT. Full training and fresh metric reproduction were not performed for this documentation update.
