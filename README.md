# News Topic Classification with DistilBERT

## Objective

Fine-tune a transformer-based model (DistilBERT) to classify news headlines from the AG News dataset into four categories: **World, Sports, Business, Sci/Tech**. Demonstrates text preprocessing, model training, evaluation, and live prediction deployment.

---

## Dataset

* **Name:** AG News
* **Source:** [Hugging Face Datasets](https://huggingface.co/datasets/sh0416/ag_news)
* **Columns:**

  * `text` — news headline
  * `label` — category (`World`, `Sports`, `Business`, `Sci/Tech`)

---

## Steps Performed

### 1. Data Loading & Preprocessing

* Loaded the AG News dataset using Hugging Face `datasets` library.
* Tokenized headlines with `DistilBERT` tokenizer.
* Subsampled for fast CPU training: **500 training samples**, **200 test samples**.

### 2. Model Training

* Loaded `distilbert-base-uncased` pre-trained model.
* Fine-tuned using Hugging Face `Trainer` for **1 epoch**.
* Disabled evaluation during training for speed on CPU.

### 3. Evaluation

* Evaluated on test set using **accuracy** and **F1-score**.
* Achieved:

  * Accuracy: **~0.89**
  * F1-score: **~0.89**

### 4. Prediction Example

```python
def predict_headline(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = model(**inputs)
    pred = outputs.logits.argmax(dim=-1).item()
    labels = ["World", "Sports", "Business", "Sci/Tech"]
    return labels[pred]

predict_headline("NASA launches new rover to Mars")
# Output: "Sci/Tech"
```

### 5. Deployment (Optional)

* Integrated with **Gradio** for live interaction:

```python
import gradio as gr

demo = gr.Interface(fn=predict_headline, inputs="text", outputs="text")
demo.launch()
```

---

## Key Insights

* DistilBERT provides **fast training** on CPU for small datasets.
* High performance (~0.89 accuracy/F1) achievable even with **1 epoch**.
* Tokenization and preprocessing are critical for transformer-based models.
* Gradio enables **interactive live predictions**.

---

## Files Included

* `News_Topic_Classifier.ipynb` — Jupyter Notebook with code, preprocessing, training, evaluation, and Gradio deployment.
* `bert-ag-news-distilbert/` — Saved model and tokenizer for inference (large model excluded from GitHub due to size).
* `metrics.json` — Evaluation metrics.

---

## References

* [AG News Dataset](https://huggingface.co/datasets/sh0416/ag_news)
* [Hugging Face Transformers Documentation](https://huggingface.co/docs/transformers)
