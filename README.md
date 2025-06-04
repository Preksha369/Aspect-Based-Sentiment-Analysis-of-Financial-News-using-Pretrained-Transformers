# Aspect-Based Sentiment Analysis on Financial News (SEntFiN Dataset)

This project explores the use of pretrained transformer models for **aspect-based sentiment analysis** on the [SEntFiN dataset](https://www.kaggle.com/datasets/ankurzing/aspect-based-sentiment-analysis-for-financial-news), which contains over 10,000 financial news headlines. Each headline is annotated with relevant aspects and sentiment labels (`positive`, `neutral`, `negative`).

Rather than training new models, we evaluate the zero-shot performance of existing **Hugging Face Transformers** to determine how well they can understand sentiment in financial news.

---

## Dataset

The dataset includes:
- `Title`: A financial news headline
- `Decisions`: A dictionary mapping each financial aspect (e.g., company names) in the headline to its corresponding sentiment

After preprocessing, we convert this into a flat format where each row corresponds to a single `(Title, Aspect, Sentiment)` entry.

---

## Models Evaluated

### 1. **Twitter-RoBERTa (cardiffnlp/twitter-roberta-base-sentiment)**
- A general-purpose sentiment model trained on Twitter data
- Requires label mapping from `label_0`, `label_1`, `label_2` to `negative`, `neutral`, `positive`
- **Accuracy on SEntFiN**: ~58%

### 2. **FinBERT (ProsusAI/finbert)**
- A domain-specific model trained on financial texts
- Directly outputs sentiment labels
- **Accuracy on SEntFiN**: ~66%
- Better balance in precision and recall across all classes

### Additional Models Explored
We also briefly tested other transformer-based models such as:
- **DistilBERT** (lightweight BERT)
- **ALBERT** (parameter-efficient BERT)
- **Multilingual BERT**
- **NLPTown BERT**

However, these models yielded lower accuracy compared to FinBERT, especially due to their general-purpose training and lack of financial domain understanding. For clarity, we focus on detailed results only for the top two models.

---

## Evaluation Metrics

- **Accuracy**: Percentage of correctly predicted sentiments
- **Classification Report**: Includes precision, recall, and F1-score for each sentiment class

---

## How to Run

1. Clone this repository
2. Install dependencies:
    ```bash
    pip install pandas scikit-learn transformers
    ```
3. Run the Jupyter/Colab notebook to evaluate predictions and see results.

---

## Key Takeaway

Domain-specific models like **FinBERT** significantly outperform general models when analyzing financial news. This highlights the importance of using pretrained models tailored to the task's context, especially in specialized fields like finance.

---


