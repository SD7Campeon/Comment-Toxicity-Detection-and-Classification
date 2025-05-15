# 🧬 Comment Toxicity Detection and Classification

[![TensorFlow](https://img.shields.io/badge/Framework-TensorFlow_2.x-orange.svg)](https://www.tensorflow.org/)
[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

This repository, curated under the aegis of **SD7Campeon**, encapsulates a **high-dimensional neurocomputational framework** for **multi-label semantic toxicity disambiguation** in unstructured textual corpora. Leveraging a **bidirectional LSTM architecture** within TensorFlow’s Keras ecosystem, this project operationalizes **contextual feature extraction** and **nonlinear discriminant analysis** to classify six orthogonal toxicity dimensions: `toxic`, `severe_toxic`, `obscene`, `threat`, `insult`, and `identity_hate`. The implementation integrates a **Gradio-based interactive inference interface** for real-time toxicity scoring.

---

## 🔬 Ontological Premise

The proliferation of deleterious linguistic constructs in digital discourse necessitates robust mechanisms for **automated toxicity detection**. This repository instantiates a **deep sequential learning paradigm** to perform **multi-label classification**, addressing the **semantic heterogeneity** of toxic expressions in user-generated content. The model is trained on the **Jigsaw Toxic Comment Classification corpus**, a benchmark dataset for multi-label text analysis.

---

## 📑 Corpus Schema

The input dataset (`train.csv`) adheres to the following structure:

| Attribute         | Semantic Role                             |
|-------------------|-------------------------------------------|
| `comment_text`    | Unstructured natural language input       |
| `toxic`           | General toxicity indicator (Bernoulli)   |
| `severe_toxic`    | High-severity toxicity flag              |
| `obscene`         | Profane language marker                 |
| `threat`          | Threat-oriented expression identifier    |
| `insult`          | Personal attack signifier                |
| `identity_hate`   | Identity-based hate speech indicator     |

> **Note**: Ensure `train.csv` resides in the project root directory prior to execution.

---

## 🛠️ Dependency Constellation

Install requisite libraries via `pip`:

```bash
pip install tensorflow pandas matplotlib scikit-learn gradio jinja2
```

---

## 🧪 Computational Pipeline

### 1. **Lexico-Semantic Preprocessing**
- **Token Vectorization**: Employs `TextVectorization` to construct a **high-capacity lexical embedding space** (MAX_FEATURES=200,000 tokens, sequence length=1,800 tokens, integer output mode).
- **TensorFlow Data Pipeline**: Utilizes `tf.data.Dataset` with **MCSHBAP** (Map, Cache, Shuffle, Batch, Prefetch) optimization for efficient data streaming.

```python
vectorizer = TextVectorization(
    max_tokens=200_000,
    output_sequence_length=1800,
    output_mode='int'
)
vectorizer.adapt(X.values)
dataset = tf.data.Dataset.from_tensor_slices((vectorized_text, y))
dataset = dataset.cache().shuffle(160_000).batch(16).prefetch(8)
```

### 2. **Neural Architecture**
The model is architected as a **sequential deep learning topology**:

```python
model = Sequential([
    Embedding(MAX_FEATURES+1, 32),
    Bidirectional(LSTM(32, activation='tanh')),
    Dense(128, activation='relu'),
    Dense(256, activation='relu'),
    Dense(128, activation='relu'),
    Dense(6, activation='sigmoid')
])
```

- **Embedding Layer**: Projects tokens into a 32-dimensional semantic manifold.
- **Bidirectional LSTM**: Captures **bidirectional contextual dependencies** via gated recurrent units with hyperbolic tangent activation.
- **Dense Layers**: Facilitate **nonlinear feature extraction** through a cascade of fully connected layers with ReLU activations.
- **Output Layer**: Emits six sigmoid-activated probabilities, enabling **independent multi-label predictions**.

### 3. **Optimization & Compilation**
- **Loss Function**: Binary Crossentropy, optimized per-label for multi-target classification.
- **Optimizer**: Adam, with adaptive learning rate dynamics.

```python
model.compile(loss='BinaryCrossentropy', optimizer='Adam')
```

### 4. **Training Regime**
- **Data Partitioning**: 70% training, 20% validation, 10% testing.
- **Training**: Single-epoch training for demonstration (extensible to multi-epoch regimes).

```python
history = model.fit(train, epochs=1, validation_data=val)
```

### 5. **Evaluation Metrics**
Metrics are computed via streaming updates:

- **Precision**: True Positives / (True Positives + False Positives)
- **Recall**: True Positives / (True Positives + False Negatives)
- **Categorical Accuracy**: Binary classification accuracy per label.

```python
pre = Precision()
re = Recall()
acc = CategoricalAccuracy()
for batch in test.as_numpy_iterator():
    X_true, y_true = batch
    yhat = model.predict(X_true)
    y_true = y_true.flatten()
    yhat = yhat.flatten()
    pre.update_state(y_true, yhat)
    re.update_state(y_true, yhat)
    acc.update_state(y_true, yhat)
print(f'Precision: {pre.result().numpy()}, Recall: {re.result().numpy()}, Accuracy: {acc.result().numpy()}')
```

---

## 🌐 Real-Time Inference

### **Single-Instance Prediction**
For isolated text scoring:

```python
input_str = vectorizer("You are utterly deplorable!")
result = model.predict(np.expand_dims(input_str, 0))
print((result > 0.5).astype(int))
```

### **Gradio Interactive Interface**
A **web-based interface** enables real-time toxicity scoring:

```python
import gradio as gr

def score_comment(comment):
    vectorized_comment = vectorizer([comment])
    results = model.predict(vectorized_comment)
    text = ''
    for idx, col in enumerate(['toxic', 'severe_toxic', 'obscene', 'threat', 'insult', 'identity_hate']):
        text += f'{col}: {results[0][idx] > 0.5}\n'
    return text

interface = gr.Interface(
    fn=score_comment,
    inputs=gr.Textbox(lines=2, placeholder="Enter comment for toxicity analysis..."),
    outputs="textbox"
)
interface.launch()
```

---

## 📈 Visualization

Training and validation loss are visualized using Matplotlib:

```python
from matplotlib import pyplot as plt
plt.figure(figsize=(8, 5))
pd.DataFrame(history.history).plot()
plt.show()
```

---

## 🧠 Advanced Methodological Constructs
- **Bidirectional Sequence Modeling**: Captures **long-range contextual dependencies** via dual-path LSTM.
- **Multi-Label Paradigm**: Independent sigmoid activations for non-mutually exclusive labels.
- **Data Pipeline Optimization**: Leverages `tf.data` for **asynchronous data prefetching** and **in-memory caching**.
- **Gradio Integration**: Facilitates **real-time human-in-the-loop evaluation**.

---

## 📦 Model Persistence

The model is serialized for reusability:

```python
model.save('toxicity.h5')
model = tf.keras.models.load_model('toxicity.h5')
```

---

## 🚀 Prospective Enhancements
- **Attention-Augmented Architectures**: Integrate **self-attention mechanisms** for enhanced contextual modeling.
- **Model Quantization**: Optimize for edge deployment using TensorFlow Lite.
- **API Deployment**: Expose inference endpoints via FastAPI or Flask.
- **Domain Adaptation**: Fine-tune embeddings on domain-specific corpora.

---

## 🧾 License

This project is licensed under the **MIT License**. See [LICENSE](LICENSE) for details.

---

## 👨‍🔬 Maintainer

**SD7Campeon**   
🌐 github.com/SD7Campeon

---

## ⭐ Contributing

Contributions are welcomed via:

1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/enhancement`).
3. Commit changes (`git commit -m "Add enhancement"`).
4. Push to the branch (`git push origin feature/enhancement`).
5. Open a Pull Request.

---

## 🧰 Quick Start

1. Clone the repository:
   ```bash
   git clone https://github.com/SD7Campeon/Comment-Toxicity-Detection-and-Classification.git
   ```
2. Install dependencies (see above).
3. Place `train.csv` in the project root.
4. Execute the pipeline script:
   ```bash
   python toxicity_classifier.py
   ```
5. Launch the Gradio interface:
   ```bash
   python -m gradio run toxicity_classifier.py
   ```