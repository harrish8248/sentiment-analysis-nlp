# ============================================================
#   Sentiment Analysis on Movie Reviews using NLP
#   Author: Harrish Sebastin A
#   Tools: Python, NLTK, Scikit-learn, Matplotlib, Seaborn
# ============================================================

import os
import re
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')
import seaborn as sns

import nltk
from nltk.corpus import movie_reviews, stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, classification_report,
                             ConfusionMatrixDisplay, confusion_matrix)
from wordcloud import WordCloud

# ── Download required NLTK data ──────────────────────────────
print("=" * 55)
print("  Sentiment Analysis on Movie Reviews — NLP Project")
print("=" * 55)

print("\n[1/6] Downloading NLTK datasets...")
nltk.download('movie_reviews', quiet=True)
nltk.download('stopwords',     quiet=True)
nltk.download('punkt',         quiet=True)
print("      Done.")

# ── Load Dataset ─────────────────────────────────────────────
print("\n[2/6] Loading dataset...")

documents = []
for category in movie_reviews.categories():
    for fileid in movie_reviews.fileids(category):
        documents.append((movie_reviews.raw(fileid), category))

df = pd.DataFrame(documents, columns=['review', 'sentiment'])
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"      Total reviews loaded : {len(df)}")
print(f"      Positive             : {(df['sentiment']=='pos').sum()}")
print(f"      Negative             : {(df['sentiment']=='neg').sum()}")

# ── Preprocess Text ───────────────────────────────────────────
print("\n[3/6] Preprocessing text...")

stop_words = set(stopwords.words('english'))
stemmer    = PorterStemmer()

def clean_text(text):
    text = re.sub(r'<.*?>',    '', text)   # remove HTML
    text = re.sub(r'[^a-zA-Z]', ' ', text) # keep letters only
    text = text.lower().split()
    text = [stemmer.stem(w) for w in text if w not in stop_words and len(w) > 2]
    return ' '.join(text)

df['cleaned'] = df['review'].apply(clean_text)
print("      Text cleaning complete.")

# ── Train / Test Split & Vectorization ───────────────────────
print("\n[4/6] Training model...")

X_train, X_test, y_train, y_test = train_test_split(
    df['cleaned'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment'])

tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
X_train_tf = tfidf.fit_transform(X_train)
X_test_tf  = tfidf.transform(X_test)

model = LogisticRegression(max_iter=1000, C=1.0)
model.fit(X_train_tf, y_train)

preds    = model.predict(X_test_tf)
accuracy = accuracy_score(y_test, preds) * 100

print(f"\n{'─'*45}")
print(f"  ✅  Model Accuracy : {accuracy:.2f}%")
print(f"{'─'*45}")
print("\n  Classification Report:")
print(classification_report(y_test, preds, target_names=['Negative', 'Positive']))

# ── Generate Visualizations ───────────────────────────────────
print("[5/6] Generating visualizations...")
os.makedirs("outputs", exist_ok=True)

# 1. Sentiment Distribution
fig, ax = plt.subplots(figsize=(7, 4))
colors = ['#E74C3C', '#2ECC71']
df['sentiment'].map({'neg': 'Negative', 'pos': 'Positive'}).value_counts().plot(
    kind='bar', ax=ax, color=colors, edgecolor='white', width=0.5)
ax.set_title('Sentiment Distribution', fontsize=14, fontweight='bold', pad=15)
ax.set_xlabel('Sentiment', fontsize=11)
ax.set_ylabel('Number of Reviews', fontsize=11)
ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
for p in ax.patches:
    ax.annotate(f'{int(p.get_height())}',
                (p.get_x() + p.get_width() / 2., p.get_height()),
                ha='center', va='bottom', fontsize=11)
plt.tight_layout()
plt.savefig('outputs/1_sentiment_distribution.png', dpi=150)
plt.close()
print("      Saved: outputs/1_sentiment_distribution.png")

# 2. Word Cloud — Positive Reviews
positive_text = ' '.join(df[df['sentiment'] == 'pos']['cleaned'])
wc = WordCloud(width=900, height=450, background_color='white',
               colormap='Greens', max_words=150).generate(positive_text)
plt.figure(figsize=(10, 5))
plt.imshow(wc, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words in Positive Reviews', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('outputs/2_wordcloud_positive.png', dpi=150)
plt.close()
print("      Saved: outputs/2_wordcloud_positive.png")

# 3. Word Cloud — Negative Reviews
negative_text = ' '.join(df[df['sentiment'] == 'neg']['cleaned'])
wc2 = WordCloud(width=900, height=450, background_color='white',
                colormap='Reds', max_words=150).generate(negative_text)
plt.figure(figsize=(10, 5))
plt.imshow(wc2, interpolation='bilinear')
plt.axis('off')
plt.title('Most Frequent Words in Negative Reviews', fontsize=14, fontweight='bold', pad=12)
plt.tight_layout()
plt.savefig('outputs/3_wordcloud_negative.png', dpi=150)
plt.close()
print("      Saved: outputs/3_wordcloud_negative.png")

# 4. Confusion Matrix
cm = confusion_matrix(y_test, preds)
fig, ax = plt.subplots(figsize=(6, 5))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'], ax=ax,
            linewidths=0.5, linecolor='gray')
ax.set_title(f'Confusion Matrix  (Accuracy: {accuracy:.2f}%)', fontsize=13, fontweight='bold', pad=12)
ax.set_xlabel('Predicted Label', fontsize=11)
ax.set_ylabel('True Label', fontsize=11)
plt.tight_layout()
plt.savefig('outputs/4_confusion_matrix.png', dpi=150)
plt.close()
print("      Saved: outputs/4_confusion_matrix.png")

# 5. Top 20 most important features
feature_names = tfidf.get_feature_names_out()
coef = model.coef_[0]
top_pos_idx = np.argsort(coef)[-20:][::-1]
top_neg_idx = np.argsort(coef)[:20]

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
ax1.barh([feature_names[i] for i in top_pos_idx[::-1]],
         [coef[i] for i in top_pos_idx[::-1]], color='#2ECC71')
ax1.set_title('Top 20 Positive Indicator Words', fontweight='bold')
ax1.set_xlabel('TF-IDF Coefficient')

ax2.barh([feature_names[i] for i in top_neg_idx[::-1]],
         [coef[i] for i in top_neg_idx[::-1]], color='#E74C3C')
ax2.set_title('Top 20 Negative Indicator Words', fontweight='bold')
ax2.set_xlabel('TF-IDF Coefficient')

plt.suptitle('Most Influential Words for Sentiment Prediction', fontsize=13, fontweight='bold', y=1.01)
plt.tight_layout()
plt.savefig('outputs/5_top_features.png', dpi=150, bbox_inches='tight')
plt.close()
print("      Saved: outputs/5_top_features.png")

# ── Live Prediction Demo ──────────────────────────────────────
print("\n[6/6] Running sample predictions...\n")

samples = [
    "This movie was absolutely fantastic! Great acting and brilliant story.",
    "Terrible film. Waste of time, awful acting and a boring plot.",
    "The cinematography was stunning and the performances were outstanding.",
    "I fell asleep halfway through. Completely dull and predictable.",
    "One of the best movies I have seen in years. Highly recommended!"
]

print(f"  {'Review':<58} {'Prediction':>12}")
print(f"  {'─'*58} {'─'*12}")
for sample in samples:
    cleaned  = clean_text(sample)
    vec      = tfidf.transform([cleaned])
    pred     = model.predict(vec)[0]
    label    = "✅ POSITIVE" if pred == 'pos' else "❌ NEGATIVE"
    preview  = sample[:55] + "..." if len(sample) > 55 else sample
    print(f"  {preview:<58} {label:>12}")

print(f"\n{'='*55}")
print(f"  Project complete! Accuracy achieved: {accuracy:.2f}%")
print(f"  All charts saved in the 'outputs/' folder.")
print(f"{'='*55}\n")
