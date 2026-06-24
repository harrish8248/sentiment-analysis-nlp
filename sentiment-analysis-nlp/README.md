# 🎬 Movie Review Sentiment Analysis using NLP & Machine Learning

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](https://opensource.org/licenses/MIT)
[![Streamlit App](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=flat&logo=streamlit&logoColor=white)](https://streamlit.io/)
[![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=flat&logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![NLTK](https://img.shields.io/badge/NLTK-NLP-blue)](https://www.nltk.org/)

An end-to-end Natural Language Processing (NLP) and Machine Learning project that classifies movie reviews as **Positive** or **Negative**. Utilizing the NLTK Movie Reviews dataset, this project implements two interfaces: a CLI training pipeline and an interactive **Streamlit web application** with a premium dark theme. The model employs custom text preprocessing, TF-IDF feature engineering, and a Logistic Regression classifier to achieve **~84% to 90% classification accuracy**.

---

## 📌 Table of Contents
1. [Key Features](#-key-features)
2. [Project Architecture & Pipeline](#-project-architecture--pipeline)
3. [Project Structure](#-project-structure)
4. [Installation & Setup](#-installation--setup)
5. [Running the Application](#-running-the-application)
6. [Model Evaluation & Results](#-model-evaluation--results)
7. [Visualizations Summary](#-visualizations-summary)
8. [Technical Interview Talking Points](#-technical-interview-talking-points)

---

## ✨ Key Features
* **Interactive Streamlit Web App**: A modern wide-layout frontend featuring custom dark UI CSS, responsive sentiment score progress bars, and template selectors.
* **Cached Model Pipelines**: Utilizes `@st.cache_resource` in the web application to download the NLTK datasets and train the Logistic Regression classifier only once on startup.
* **Robust Text Preprocessing**: Custom cleaning function including HTML stripping, non-alphabetic filtering, lowercasing, stopword removal, and stemming using `PorterStemmer`.
* **Advanced Vectorization**: Implements TF-IDF representation extracting up to 15,000 unigram and bigram features (`ngram_range=(1,2)`).
* **High-Performance Classifier**: Leverages Logistic Regression with L2 regularization to produce clean, interpretable, and highly generalizable predictions.
* **Auto-generated Visualizations**: Automatically generates and saves 5 diagnostic plots to the `outputs/` folder.

---

## 🏗️ Project Architecture & Pipeline

```
[ NLTK Dataset ] ➔ [ clean_text() ] ➔ [ TF-IDF unigram+bigram ] ➔ [ Logistic Regression ] ➔ [ Metrics & Visualizations ]
```

1. **Text Preprocessing**: Normalizes inputs (strips HTML, regex filters to `[a-zA-Z]`), lowercases, tokenizes, discards stopwords, and stems words using the `PorterStemmer` to handle lexical variations.
2. **Feature Engineering**: Converts cleaned strings into numerical TF-IDF vectors. Bigrams are used to capture contextual negation (e.g., "not bad", "never good").
3. **Training & Validation**: Splits the dataset 80/20 with stratification to maintain class balance in training and testing phases.
4. **Analysis & Diagnostics**: Computes precision, recall, and F1-scores, and extracts the classifier's coefficients to determine the most influential positive/negative words.

---

## 📁 Project Structure

```
sentiment-analysis-nlp/
├── app.py                   # Streamlit web app interface (interactive UI)
├── sentiment_analysis.py    # Original model training and CLI script (untouched)
├── requirements.txt         # Package dependencies (updated with Streamlit)
├── README.md                # Project documentation
│
└── outputs/                 # Automatically created folder containing plots
    ├── 1_sentiment_distribution.png  # Class balance verification
    ├── 2_wordcloud_positive.png      # Top words in positive reviews
    ├── 3_wordcloud_negative.png      # Top words in negative reviews
    ├── 4_confusion_matrix.png        # Classifier validation heatmap
    └── 5_top_features.png            # Feature importance / TF-IDF coefficients
```

---

## ⚙️ Installation & Setup

### Prerequisites
* Python 3.8 or higher installed on your system.

### Step 1: Clone the Repository
```bash
git clone https://github.com/harrish8248/sentiment-analysis-nlp.git
cd sentiment-analysis-nlp
```

### Step 2: Set Up Virtual Environment
Create and activate a virtual environment to isolate project dependencies:

**On Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```

**On macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

---

## 🚀 Running the Application

This project supports two execution modes:

### 1. Interactive Streamlit Web App (Recommended)
Launch the interactive web portal:
```bash
streamlit run app.py
```
Open your browser and navigate to `http://localhost:8501`. You can type custom reviews, click quick-fill templates, and view the diagnostic visualization dashboard directly in the app.

### 2. Command Line Interface (CLI) Pipeline
Run the training pipeline script to print evaluation metrics and regenerate diagnostic plots:
```bash
python sentiment_analysis.py
```

---

## 📊 Model Evaluation & Results

The Logistic Regression model is evaluated on a stratified test set (20% of the total dataset).

### Performance Metrics
* **Accuracy**: **~84.0% - 90.0%** (varies slightly based on stratified split variance)
* **Precision/Recall Balance**: Shows equal performance across both positive and negative classes due to the perfectly balanced corpus.

```
              precision    recall  f1-score   support

    Negative       0.83      0.85      0.84       200
    Positive       0.85      0.83      0.84       200

    accuracy                           0.84       400
   macro avg       0.84      0.84      0.84       400
weighted avg       0.84      0.84      0.84       400
```

---

## 🎨 Visualizations Summary

All plots are generated automatically and saved to the `outputs/` directory:
* **Sentiment Distribution**: Verifies class parity (1,000 positive / 1,000 negative reviews).
* **Positive/Negative Word Clouds**: Highlight high-frequency sentiment terms (e.g., "brilliant", "great" vs. "bad", "waste").
* **Confusion Matrix Heatmap**: Displays exact counts of True Positives, True Negatives, False Positives, and False Negatives to diagnose classifier bias.
* **Top Features Bar Chart**: Plots the top 20 positive and negative words sorted by their Logistic Regression beta coefficients.

---

## 💡 Technical Interview Talking Points

1. **Why Logistic Regression over Deep Learning (e.g., LSTMs, Transformers)?**
   * *Answer*: For a dataset of this size (2,000 samples), deep learning models would easily overfit without extensive pretraining. Logistic Regression acts as a very strong, highly interpretable baseline, trains in sub-seconds, and runs efficiently without GPU acceleration.
2. **Why use TF-IDF with Bigrams instead of simple Bag-of-Words (CountVectorizer)?**
   * *Answer*: TF-IDF penalizes corpus-wide common words (like "movie", "film"), accentuating terms that characterize specific documents. Bigrams (`ngram_range=(1,2)`) help capture local phrases like "not good" or "waste of", which are critical for resolving sentiment polarity.
3. **How does Stemming compare to Lemmatization in this context?**
   * *Answer*: Stemming (`PorterStemmer`) uses heuristic rules to slice suffixes off words, which is fast and reduces vocabulary dimensionality. Lemmatization uses morphological analysis and a dictionary (like WordNet) to find the base form. For classification tasks, stemming is often sufficient and faster.

---

## 📄 License
This project is licensed under the MIT License - see the LICENSE file for details.
