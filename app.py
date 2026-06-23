from flask import Flask, jsonify, request, render_template_string
from flask_cors import CORS
import pandas as pd
import numpy as np
import re
import os
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                             f1_score, confusion_matrix, classification_report)
from collections import Counter
import string

app = Flask(__name__)
CORS(app)


app_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(app_dir, 'spam.csv')

df = pd.read_csv(csv_path, encoding='latin1',
                 usecols=[0, 1], names=['label', 'message'], header=0)
df.dropna(subset=['message'], inplace=True)
df['label_num'] = (df['label'] == 'spam').astype(int)

def clean(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\d+', '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return text.strip()

df['clean'] = df['message'].apply(clean)
df['word_count']  = df['message'].apply(lambda x: len(x.split()))
df['char_count']  = df['message'].apply(len)
df['has_url']     = df['message'].apply(lambda x: int(bool(re.search(r'http|www', x, re.I))))
df['excl_count']  = df['message'].apply(lambda x: x.count('!'))
df['upper_ratio'] = df['message'].apply(
    lambda x: sum(1 for c in x if c.isupper()) / max(len(x), 1))

X_train, X_test, y_train, y_test = train_test_split(
    df['clean'], df['label_num'], test_size=0.2, random_state=42, stratify=df['label_num'])

vec   = TfidfVectorizer(max_features=5000, stop_words='english')
X_tr  = vec.fit_transform(X_train)
X_te  = vec.transform(X_test)

model = MultinomialNB()
model.fit(X_tr, y_train)
y_pred = model.predict(X_te)

feature_names = np.array(vec.get_feature_names_out())
spam_idx  = list(model.classes_).index(1)
ham_idx   = list(model.classes_).index(0)
spam_log  = model.feature_log_prob_[spam_idx]
ham_log   = model.feature_log_prob_[ham_idx]
diff       = spam_log - ham_log
top_spam_idx = diff.argsort()[-20:][::-1]
top_ham_idx  = diff.argsort()[:20]

def word_freq(series, top=20):
    words = ' '.join(series).split()
    return Counter(words).most_common(top)


@app.route('/api/stats', methods=['GET'])
def stats():
    total   = len(df)
    n_spam  = int((df['label'] == 'spam').sum())
    n_ham   = int((df['label'] == 'ham').sum())


    spam_df = df[df['label'] == 'spam']
    ham_df  = df[df['label'] == 'ham']

    bins   = list(range(0, 201, 20)) + [9999]
    labels = [f'{b}-{bins[i+1] if bins[i+1]!=9999 else "200+"}' for i, b in enumerate(bins[:-1])]

    spam_hist = pd.cut(spam_df['word_count'], bins=bins, labels=labels).value_counts().sort_index()
    ham_hist  = pd.cut(ham_df['word_count'],  bins=bins, labels=labels).value_counts().sort_index()

    # monthly-style fake trend (message index bucketed into 20 buckets)
    df['bucket'] = pd.cut(df.index, bins=20, labels=False)
    trend = df.groupby(['bucket', 'label']).size().unstack(fill_value=0).reset_index()

    return jsonify({
        'total':  total,
        'spam':   n_spam,
        'ham':    n_ham,
        'spam_pct': round(n_spam / total * 100, 1),
        'ham_pct':  round(n_ham  / total * 100, 1),

     
        'length_dist': {
            'labels': labels,
            'spam':   spam_hist.tolist(),
            'ham':    ham_hist.tolist(),
        },

      
        'volume_trend': {
            'batches': trend['bucket'].tolist(),
            'spam':    trend.get('spam', pd.Series([0]*20)).tolist(),
            'ham':     trend.get('ham',  pd.Series([0]*20)).tolist(),
        },

     
        'top_spam_words': word_freq(spam_df['clean']),
        'top_ham_words':  word_freq(ham_df['clean']),

  
        'top_spam_features': [
            {'word': feature_names[i], 'score': round(float(diff[i]), 3)}
            for i in top_spam_idx
        ],
        'top_ham_features': [
            {'word': feature_names[i], 'score': round(float(-diff[i]), 3)}
            for i in top_ham_idx
        ],

     
        'avg_word_count': {
            'spam': round(float(spam_df['word_count'].mean()), 1),
            'ham':  round(float(ham_df['word_count'].mean()),  1),
        },
        'avg_char_count': {
            'spam': round(float(spam_df['char_count'].mean()), 1),
            'ham':  round(float(ham_df['char_count'].mean()),  1),
        },
    })


@app.route('/api/model', methods=['GET'])
def model_metrics():
    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=['ham', 'spam'], output_dict=True)
    return jsonify({
        'accuracy':  round(accuracy_score(y_test, y_pred) * 100, 2),
        'precision': round(precision_score(y_test, y_pred) * 100, 2),
        'recall':    round(recall_score(y_test, y_pred) * 100, 2),
        'f1':        round(f1_score(y_test, y_pred) * 100, 2),
        'confusion_matrix': cm.tolist(),
        'report':    report,
        'train_size': len(X_train),
        'test_size':  len(X_test),
    })


@app.route('/api/predict', methods=['POST'])
def predict():
    data    = request.get_json()
    message = data.get('message', '').strip()
    if not message:
        return jsonify({'error': 'Empty message'}), 400

    cleaned  = clean(message)
    vec_msg  = vec.transform([cleaned])
    proba    = model.predict_proba(vec_msg)[0]
    label    = 'spam' if proba[1] >= 0.5 else 'ham'

  
    tfidf_vals  = vec_msg.toarray()[0]
    nonzero_idx = tfidf_vals.nonzero()[0]
    word_scores = [(feature_names[i], round(float(tfidf_vals[i] * diff[i]), 4))
                   for i in nonzero_idx]
    word_scores.sort(key=lambda x: x[1], reverse=True)

    return jsonify({
        'label':        label,
        'confidence':   round(float(max(proba)) * 100, 2),
        'spam_prob':    round(float(proba[1]) * 100, 2),
        'ham_prob':     round(float(proba[0]) * 100, 2),
        'word_count':   len(message.split()),
        'char_count':   len(message),
        'top_features': word_scores[:10],
    })



@app.route('/')
def dashboard():
    """Serve the main dashboard"""
    with open(os.path.join(app_dir, 'dashboard.html'), 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content


@app.route('/dashboard')
def dashboard_page():
    """Alternative route for dashboard"""
    with open(os.path.join(app_dir, 'dashboard.html'), 'r', encoding='utf-8') as f:
        html_content = f.read()
    return html_content


if __name__ == '__main__':
    app.run(debug=True, port=5050)
