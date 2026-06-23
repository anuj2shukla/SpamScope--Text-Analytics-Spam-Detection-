# SpamScope--Text-Analytics-Spam-Detection-
A full-stack text analytics system for classifying SMS messages as spam or legitimate using Naive Bayes machine learning, with a live interactive dashboard for exploration and real-time inference.

Overview
SpamScope is built on the UCI SMS Spam Collection dataset — 5,572 real-world SMS messages labelled as spam or ham (legitimate). The system trains a Multinomial Naive Bayes classifier with TF-IDF feature extraction entirely at startup, then exposes the results through a Flask REST API consumed by a browser-based analytics dashboard.

The project is designed to answer three questions a stakeholder would actually care about:

What does the data look like? — Distribution of spam vs. ham, message length patterns, most frequent terms by class.
How well does the model perform? — Accuracy, precision, recall, F1, confusion matrix, and per-class breakdown.
Does it work on new messages? — A live classifier that analyses any text in real time and explains which words drove the decision.


Features

Overview page — KPI strip with total messages, spam count, ham count, and model accuracy; insight callouts on average message length; frequency term clouds for spam and ham; class distribution doughnut chart.
Charts page — Bar chart of message length distribution (word-count buckets, spam vs. ham); line graph of volume trends across dataset batches; horizontal bar chart of the top discriminative TF-IDF features ranked by log-probability difference.
Model page — Four metric tiles (Accuracy, Precision, Recall, F1) with animated SVG progress rings; colour-coded confusion matrix with true/false positive and negative counts; grouped bar chart of per-class precision, recall, and F1.
Classifier page — Real-time message input with a live keyword-risk meter; sample messages to try; probability bars for spam and ham; per-feature contribution bars showing which words influenced the decision; session history table logging every classification.


Machine Learning Pipeline


Text cleaning: lowercasing, URL removal, digit stripping, punctuation removal.
TF-IDF vectorisation with a 5,000-feature vocabulary and English stop-word filtering.
Multinomial Naive Bayes trained on an 80/20 stratified train-test split.
Log-probability difference scores used to rank and explain feature importance per class.
TF-IDF vectorisation with a 5,000-feature vocabulary and English stop-word filtering.
Multinomial Naive Bayes trained on an 80/20 stratified train-test split.
Log-probability difference scores used to rank and explain feature importance per class.
