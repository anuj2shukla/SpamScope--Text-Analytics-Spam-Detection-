 SpamScope — Intelligent Spam Detection Analytics

> A full-stack text analytics system for classifying SMS messages as spam or legitimate using Naive Bayes machine learning, with a live interactive dashboard for exploration and real-time inference.

---

## Overview

SpamScope is built on the **UCI SMS Spam Collection dataset** — 5,572 real-world SMS messages labelled as spam or ham (legitimate). The system trains a **Multinomial Naive Bayes classifier** with **TF-IDF feature extraction** entirely at startup, then exposes the results through a Flask REST API consumed by a browser-based analytics dashboard.

The project is designed to answer three questions a stakeholder would actually care about:

1. **What does the data look like?** — Distribution of spam vs. ham, message length patterns, most frequent terms by class.
2. **How well does the model perform?** — Accuracy, precision, recall, F1, confusion matrix, and per-class breakdown.
3. **Does it work on new messages?** — A live classifier that analyses any text in real time and explains which words drove the decision.

---

## Features

### Analytics Dashboard

- **Overview page** — KPI strip with total messages, spam count, ham count, and model accuracy; insight callouts on average message length; frequency term clouds for spam and ham; class distribution doughnut chart.
- **Charts page** — Bar chart of message length distribution (word-count buckets, spam vs. ham); line graph of volume trends across dataset batches; horizontal bar chart of the top discriminative TF-IDF features ranked by log-probability difference.
- **Model page** — Four metric tiles (Accuracy, Precision, Recall, F1) with animated SVG progress rings; colour-coded confusion matrix with true/false positive and negative counts; grouped bar chart of per-class precision, recall, and F1.
- **Classifier page** — Real-time message input with a live keyword-risk meter; sample messages to try; probability bars for spam and ham; per-feature contribution bars showing which words influenced the decision; session history table logging every classification.

### Machine Learning Pipeline

- Text cleaning: lowercasing, URL removal, digit stripping, punctuation removal.
- TF-IDF vectorisation with a 5,000-feature vocabulary and English stop-word filtering.
- Multinomial Naive Bayes trained on an 80/20 stratified train-test split.
- Log-probability difference scores used to rank and explain feature importance per class.

---

## Performance

| Metric    | Score  |
|-----------|--------|
| Accuracy  | 96.59% |
| Precision | 99.12% |
| Recall    | 85.50% |
| F1 Score  | 91.80% |

Evaluated on a held-out test set of 1,115 messages (20% of the corpus, stratified by class).

---

## Tech Stack

| Layer     | Technology                                      |
|-----------|-------------------------------------------------|
| Backend   | Python 3, Flask, flask-cors                     |
| ML        | scikit-learn (MultinomialNB, TfidfVectorizer)   |
| Data      | pandas, NumPy                                   |
| Frontend  | Vanilla HTML / CSS / JavaScript                 |
| Charts    | Chart.js 4                                      |
| Dataset   | UCI SMS Spam Collection (`spam.csv`)            |

---

## Project Structure

```
spam-detection/
├── app.py            # Flask backend — model training + REST API
├── dashboard.html    # Frontend analytics dashboard
├── spam.csv          # SMS Spam Collection dataset
└── README.md
```


## Dataset

The **SMS Spam Collection** is a public dataset of 5,572 tagged SMS messages collected for mobile phone spam research.

| Class | Count | Share  |
|-------|-------|--------|
| Ham   | 4,825 | 86.6%  |
| Spam  | 747   | 13.4%  |

The dataset is heavily imbalanced toward legitimate messages — consistent with real inbox distributions. The model handles this well due to Naive Bayes's strong performance on text classification tasks with skewed class priors.

---

## How It Works

1. **At startup**, `app.py` loads and cleans the dataset, fits a TF-IDF vectoriser on the training split, and trains a Multinomial Naive Bayes model. Feature importance scores are precomputed as the log-probability difference between the spam and ham class distributions for every vocabulary term.

2. **At inference**, a new message is cleaned using the same pipeline, vectorised with the fitted TF-IDF model, and passed to the classifier. The posterior probabilities for spam and ham are returned alongside the TF-IDF weights multiplied by the feature importance scores — giving a ranked explanation of which words most influenced the prediction.

3. **The dashboard** fetches data from the three API endpoints on page load and renders all charts and metrics client-side using Chart.js.

---

## License

This project uses the SMS Spam Collection dataset, made available for research and educational use. The dataset is credited to Tiago A. Almeida and José María Gómez Hidalgo.
