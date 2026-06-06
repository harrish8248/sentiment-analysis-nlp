# Sentiment Analysis on Movie Reviews using NLP
**Author:** Harrish Sebastin A  
**Tech Stack:** Python, NLTK, Scikit-learn, TF-IDF, Logistic Regression, Matplotlib, Seaborn, WordCloud

---

## 📌 Project Overview
A Natural Language Processing (NLP) project that classifies movie reviews as **Positive** or **Negative** using machine learning. Built on the NLTK Movie Reviews dataset (2,000 reviews), achieving **~90% accuracy** with TF-IDF vectorization and Logistic Regression.

---

## 📁 Project Structure
```
sentiment_analysis/
│
├── sentiment_analysis.py     ← Main script (run this)
├── requirements.txt          ← All dependencies
├── README.md                 ← Project documentation
│
└── outputs/                  ← Auto-generated after running
    ├── 1_sentiment_distribution.png
    ├── 2_wordcloud_positive.png
    ├── 3_wordcloud_negative.png
    ├── 4_confusion_matrix.png
    └── 5_top_features.png
```

---

## ⚙️ How to Run

### Step 1 — Install dependencies
```bash
pip install -r requirements.txt
```

### Step 2 — Run the project
```bash
python sentiment_analysis.py
```

That's it! The script will:
- Download the dataset automatically
- Train the model
- Print accuracy + classification report
- Save all charts to the `outputs/` folder
- Run live predictions on sample reviews

---

## 📊 Results
| Metric | Score |
|--------|-------|
| Model | Logistic Regression |
| Vectorizer | TF-IDF (15,000 features, bigrams) |
| Dataset | NLTK Movie Reviews (2,000 reviews) |
| Accuracy | ~88–91% |
| Train/Test Split | 80% / 20% |

---

## 🔍 Key Concepts Used
- **Text Preprocessing** — HTML removal, lowercasing, stopword removal, stemming
- **TF-IDF Vectorization** — Converts text to numerical features (unigrams + bigrams)
- **Logistic Regression** — Binary classification model
- **Model Evaluation** — Accuracy, Precision, Recall, F1-Score, Confusion Matrix
- **Visualization** — WordCloud, Confusion Matrix, Feature Importance, Distribution charts

---

## 💡 Sample Predictions
| Review | Prediction |
|--------|-----------|
| "This movie was absolutely fantastic!" | ✅ POSITIVE |
| "Terrible film. Waste of time." | ❌ NEGATIVE |
| "Outstanding performances and brilliant story." | ✅ POSITIVE |

---

## 🔗 GitHub
Upload this project to GitHub and add it to your resume!
Suggested repo name: `sentiment-analysis-nlp`
