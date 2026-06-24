import streamlit as st
import os
import re
import pandas as pd
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Movie Review Sentiment Analyzer",
    page_icon="🎬",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Inject custom CSS for a premium dark UI with glassmorphism
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    html, body, [class*="css"], .stMarkdown {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    /* Background and Layout styling */
    .stApp {
        background: linear-gradient(135deg, #0b0f19 0%, #111827 50%, #1e1b4b 100%);
        color: #f3f4f6;
    }
    
    /* Premium Glass Cards */
    .glass-card {
        background: rgba(17, 24, 39, 0.6);
        backdrop-filter: blur(12px);
        -webkit-backdrop-filter: blur(12px);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 20px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
    }
    
    /* Title Glow */
    .glow-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #a78bfa 0%, #f472b6 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 5px;
        text-shadow: 0 0 30px rgba(167, 139, 250, 0.2);
    }
    
    .glow-subtitle {
        font-size: 1.1rem;
        color: #9ca3af;
        margin-bottom: 25px;
    }
    
    /* Section Headers */
    .section-header {
        font-size: 1.3rem;
        font-weight: 700;
        color: #e2e8f0;
        border-bottom: 1px solid rgba(255, 255, 255, 0.1);
        padding-bottom: 8px;
        margin-bottom: 15px;
    }
    
    /* Buttons Customization */
    div.stButton > button {
        border-radius: 12px;
        font-weight: 600;
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        background: rgba(255, 255, 255, 0.05);
        color: #e5e7eb;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    
    div.stButton > button:hover {
        transform: translateY(-2px);
        background: linear-gradient(90deg, #6366f1 0%, #8b5cf6 100%);
        color: white;
        border-color: transparent;
        box-shadow: 0 4px 15px rgba(99, 102, 241, 0.4);
    }
    
    /* Text Area Styling */
    div.stTextArea textarea {
        background: rgba(15, 23, 42, 0.6);
        border: 1px solid rgba(255, 255, 255, 0.1);
        color: #f3f4f6;
        border-radius: 12px;
    }
    
    div.stTextArea textarea:focus {
        border-color: #8b5cf6;
        box-shadow: 0 0 0 2px rgba(139, 92, 246, 0.2);
    }
    
    /* Sentiment Result Containers */
    .result-container {
        border-radius: 12px;
        padding: 20px;
        margin-top: 20px;
        text-align: center;
        transition: all 0.5s ease;
    }
    
    .result-positive {
        background: rgba(16, 185, 129, 0.1);
        border: 2px solid #10b981;
        color: #10b981;
        box-shadow: 0 0 20px rgba(16, 185, 129, 0.2);
    }
    
    .result-negative {
        background: rgba(239, 68, 68, 0.1);
        border: 2px solid #ef4444;
        color: #ef4444;
        box-shadow: 0 0 20px rgba(239, 68, 68, 0.2);
    }
    
    .result-title {
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 5px;
    }
    
    .result-score {
        font-size: 1.2rem;
        font-weight: 500;
        color: #f3f4f6;
    }
    
    /* Metrics display in Sidebar */
    .metric-box {
        background: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.05);
        border-radius: 10px;
        padding: 12px;
        margin-bottom: 10px;
    }
    
    .metric-label {
        font-size: 0.8rem;
        color: #9ca3af;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    
    .metric-value {
        font-size: 1.2rem;
        font-weight: 700;
        color: #a78bfa;
    }
    
    /* Footer Styling */
    .custom-footer {
        text-align: center;
        padding: 25px 0;
        font-size: 0.85rem;
        color: #6b7280;
        border-top: 1px solid rgba(255, 255, 255, 0.05);
        margin-top: 50px;
    }
</style>
""", unsafe_allow_html=True)

# ----------------- ML PIPELINE CACHED RESOURCE -----------------
@st.cache_resource
def load_and_train_pipeline():
    import nltk
    # Download resources
    nltk.download('movie_reviews', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('punkt', quiet=True)
    
    from nltk.corpus import movie_reviews
    
    # Load 2000 reviews
    documents = []
    for category in movie_reviews.categories():
        for fileid in movie_reviews.fileids(category):
            documents.append((movie_reviews.raw(fileid), category))
            
    df = pd.DataFrame(documents, columns=['review', 'sentiment'])
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    
    # Preprocess Text logic (stemming, stopwords, regex)
    from nltk.corpus import stopwords
    from nltk.stem import PorterStemmer
    
    stop_words = set(stopwords.words('english'))
    stemmer = PorterStemmer()
    
    def clean_text_local(text):
        text = re.sub(r'<.*?>', '', text)   # strip HTML
        text = re.sub(r'[^a-zA-Z]', ' ', text) # keep only letters
        text = text.lower().split()
        text = [stemmer.stem(w) for w in text if w not in stop_words and len(w) > 2]
        return ' '.join(text)
        
    df['cleaned'] = df['review'].apply(clean_text_local)
    
    # TF-IDF Vectorizer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score
    
    X_train, X_test, y_train, y_test = train_test_split(
        df['cleaned'], df['sentiment'], test_size=0.2, random_state=42, stratify=df['sentiment']
    )
    
    tfidf = TfidfVectorizer(max_features=15000, ngram_range=(1, 2))
    X_train_tf = tfidf.fit_transform(X_train)
    X_test_tf = tfidf.transform(X_test)
    
    # Logistic Regression
    model = LogisticRegression(max_iter=1000, C=1.0)
    model.fit(X_train_tf, y_train)
    
    preds = model.predict(X_test_tf)
    acc = accuracy_score(y_test, preds) * 100
    
    return model, tfidf, acc, len(df), clean_text_local

# Load model pipeline
with st.spinner("Initializing NLP pipeline and training model..."):
    model, tfidf, accuracy, dataset_size, clean_text_fn = load_and_train_pipeline()

# ----------------- SIDEBAR CONFIGURATION -----------------
st.sidebar.markdown('<div class="sidebar-title">⚙️ Model Insights</div>', unsafe_allow_html=True)

metrics = [
    ("Model Accuracy", f"{accuracy:.2f}%"),
    ("Dataset Size", f"{dataset_size:,} reviews"),
    ("Algorithm", "Logistic Regression"),
    ("Vectorizer", "TF-IDF (1, 2) Grams"),
    ("Developer Name", "Harrish Sebastin A")
]

for label, val in metrics:
    st.sidebar.markdown(f"""
    <div class="metric-box">
        <div class="metric-label">{label}</div>
        <div class="metric-value">{val}</div>
    </div>
    """, unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.info(
    "This classifier was trained on the movie reviews corpus from the NLTK library. "
    "It uses Porter Stemming and TF-IDF weighting to extract robust, generalizable signals."
)

# ----------------- MAIN LAYOUT -----------------
col_main, col_right = st.columns([1.1, 0.9])

with col_main:
    st.markdown('<div class="glow-title">🎬 Movie Review Sentiment Analyzer</div>', unsafe_allow_html=True)
    st.markdown('<div class="glow-subtitle">Instant sentiment classification powered by Logistic Regression and NLTK</div>', unsafe_allow_html=True)
    
    # Session state initialization for text input and predictions
    if 'review_input' not in st.session_state:
        st.session_state.review_input = ""
    if 'last_prediction' not in st.session_state:
        st.session_state.last_prediction = None
        
    # Helper to set text and clear prediction
    def set_example_text(text):
        st.session_state.review_input = text
        st.session_state.last_prediction = None
        
    st.markdown('<div class="section-header">💡 Try an Example Review</div>', unsafe_allow_html=True)
    
    # 6 Example Review Buttons (3 positive, 3 negative)
    pos_reviews = [
        ("👍 Masterpiece", "An absolute masterpiece! The acting was stellar, the cinematography was breathtaking, and the plot was incredibly engaging from start to finish. Highly recommended!"),
        ("👍 Brilliant", "Breathtakingly beautiful and emotional. A rare cinematic gem that combines a brilliant script with outstanding performances by the entire cast."),
        ("👍 Heartwarming", "A wonderful film that is both funny and deeply moving. The characters are rich and authentic, making this an unforgettable movie experience.")
    ]
    
    neg_reviews = [
        ("👎 Complete Disaster", "A complete disaster. The plot was full of holes, the acting was wooden and unconvincing, and I fell asleep halfway through. A waste of time and money."),
        ("👎 Horrible Writing", "Horrible writing and slow, boring pacing. It felt like a cheap cash grab with zero emotional depth or coherent storyline."),
        ("👎 Disappointing", "Extremely disappointing. Despite a talented cast, the movie is predictable, dull, and fails to deliver any real excitement or character development.")
    ]
    
    st.write("Positive Templates:")
    p_col1, p_col2, p_col3 = st.columns(3)
    with p_col1:
        st.button(pos_reviews[0][0], on_click=set_example_text, args=(pos_reviews[0][1],), use_container_width=True)
    with p_col2:
        st.button(pos_reviews[1][0], on_click=set_example_text, args=(pos_reviews[1][1],), use_container_width=True)
    with p_col3:
        st.button(pos_reviews[2][0], on_click=set_example_text, args=(pos_reviews[2][1],), use_container_width=True)
        
    st.write("Negative Templates:")
    n_col1, n_col2, n_col3 = st.columns(3)
    with n_col1:
        st.button(neg_reviews[0][0], on_click=set_example_text, args=(neg_reviews[0][1],), use_container_width=True)
    with n_col2:
        st.button(neg_reviews[1][0], on_click=set_example_text, args=(neg_reviews[1][1],), use_container_width=True)
    with n_col3:
        st.button(neg_reviews[2][0], on_click=set_example_text, args=(neg_reviews[2][1],), use_container_width=True)
        
    st.markdown('<div class="section-header">🔍 Custom Analysis</div>', unsafe_allow_html=True)
    
    # Text area (value bound to st.session_state.review_input)
    review_text = st.text_area(
        "Type or paste your movie review below:",
        value=st.session_state.review_input,
        placeholder="Enter movie review here...",
        height=160,
        key="review_input"
    )
    
    analyze_clicked = st.button("Analyze Sentiment", type="primary", use_container_width=True)
    
    # Trigger prediction on button click
    if analyze_clicked and review_text.strip():
        cleaned_text = clean_text_fn(review_text)
        vectorized_text = tfidf.transform([cleaned_text])
        prediction = model.predict(vectorized_text)[0]
        probabilities = model.predict_proba(vectorized_text)[0]
        
        pos_idx = list(model.classes_).index('pos')
        neg_idx = list(model.classes_).index('neg')
        
        pos_prob = probabilities[pos_idx]
        neg_prob = probabilities[neg_idx]
        
        if prediction == 'pos':
            label = "POSITIVE"
            confidence = pos_prob * 100
        else:
            label = "NEGATIVE"
            confidence = neg_prob * 100
            
        st.session_state.last_prediction = {
            'label': label,
            'confidence': confidence,
            'pos_prob': pos_prob,
            'neg_prob': neg_prob
        }
    
    # Display the last prediction if it exists
    if st.session_state.last_prediction:
        pred_data = st.session_state.last_prediction
        label = pred_data['label']
        confidence = pred_data['confidence']
        pos_prob = pred_data['pos_prob']
        
        is_pos = label == "POSITIVE"
        card_class = "result-positive" if is_pos else "result-negative"
        emoji = "✨" if is_pos else "⚠️"
        
        st.markdown(f"""
        <div class="result-container {card_class}">
            <div class="result-title">{emoji} {label} sentiment</div>
            <div class="result-score">Confidence Score: {confidence:.2f}%</div>
        </div>
        """, unsafe_allow_html=True)
        
        st.write("")
        st.write("**Sentiment Probability breakdown:**")
        
        # Dual colored indicators
        st.progress(pos_prob)
        col_lbl_neg, col_lbl_pos = st.columns([1, 1])
        col_lbl_neg.markdown(f"⬅️ **Negative Probability**: {(1-pos_prob)*100:.1f}%")
        col_lbl_pos.markdown(f"<div style='text-align: right;'>**Positive Probability**: {pos_prob*100:.1f}% ➡️</div>", unsafe_allow_html=True)

with col_right:
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📘 Project Overview</div>', unsafe_allow_html=True)
    st.write(
        "This Streamlit application exposes a trained **Logistic Regression** classifier "
        "built on TF-IDF textual features. The pipeline transforms raw movie reviews into "
        "predictive scores by identifying vocabulary features correlated with positive/negative sentiments."
    )
    
    st.markdown('<div class="section-header">⚙️ Machine Learning Pipeline</div>', unsafe_allow_html=True)
    pipeline_steps = [
        "**NLTK Movie Reviews**: 2,000 documents (1,000 pos / 1,000 neg)",
        "**Text Preprocessing**: Lowercasing, HTML stripping, non-letter filtering, stopword removal, and Porter Stemming",
        "**TF-IDF Vectorization**: Extracts up to 15,000 unigram/bigram features (`ngram_range=(1,2)`)",
        "**Logistic Regression**: L2 Regularization, `C=1.0`, `max_iter=1000`",
        "**Split Strategy**: 80% train, 20% validation split, stratified & random_state=42"
    ]
    for step in pipeline_steps:
        st.write(f"- {step}")
        
    st.markdown('<div class="section-header">🛠️ Tech Stack & Architecture</div>', unsafe_allow_html=True)
    st.write("**Frontend:** Streamlit, Custom HTML5/CSS3")
    st.write("**Backend/NLP:** Scikit-Learn, NLTK, PorterStemmer, TF-IDF")
    st.write("**Data Manipulation:** Pandas, NumPy")
    st.write("**Visualizations:** Matplotlib, Seaborn, Wordcloud")
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Automatically read charts from outputs folder if they exist
    st.markdown('<div class="glass-card">', unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Diagnostic Visualizations</div>', unsafe_allow_html=True)
    
    if os.path.exists("outputs/1_sentiment_distribution.png"):
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "Distribution", "Positive Words", "Negative Words", "Confusion Matrix", "Influential Words"
        ])
        with tab1:
            st.image("outputs/1_sentiment_distribution.png", use_container_width=True)
        with tab2:
            st.image("outputs/2_wordcloud_positive.png", use_container_width=True)
        with tab3:
            st.image("outputs/3_wordcloud_negative.png", use_container_width=True)
        with tab4:
            st.image("outputs/4_confusion_matrix.png", use_container_width=True)
        with tab5:
            st.image("outputs/5_top_features.png", use_container_width=True)
    else:
        st.info("📊 Plots will display here once the training visualization script completes.")
    st.markdown('</div>', unsafe_allow_html=True)

# ----------------- FOOTER -----------------
st.markdown(f"""
<div class="custom-footer">
    🎬 Movie Review Sentiment Analyzer • Developed by Harrish Sebastin A
    <br>
    NLP and Machine Learning demonstration project • Python 3.8+
</div>
""", unsafe_allow_html=True)
