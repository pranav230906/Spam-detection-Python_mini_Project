# Placeholder for pipeline_builder.py
# pipeline_builder.py
import re
import html
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression

# Optional nltk
try:
    import nltk
    from nltk.stem import WordNetLemmatizer
    from nltk.corpus import stopwords
    NLTK_AVAILABLE = True
    _WNL = WordNetLemmatizer()
    try:
        STOPWORDS = set(stopwords.words('english'))
    except Exception:
        STOPWORDS = set()
except Exception:
    NLTK_AVAILABLE = False
    STOPWORDS = set()

def clean_text(s: str) -> str:
    # Basic cleaning: lowercase, remove HTML, urls, non-alphanum (keep spaces)
    s = html.unescape(s)
    s = s.lower()
    s = re.sub(r'http\S+|www\.\S+', ' ', s)
    s = re.sub(r'<[^>]+>', ' ', s)
    s = re.sub(r'[^a-z0-9\s]', ' ', s)
    s = re.sub(r'\s+', ' ', s).strip()
    return s

def simple_tokenizer(text):
    # fallback tokenizer
    tokens = re.findall(r'\b[a-z0-9]+\b', text.lower())
    if STOPWORDS:
        tokens = [t for t in tokens if t not in STOPWORDS]
    if NLTK_AVAILABLE:
        tokens = [_WNL.lemmatize(t) for t in tokens]
    return tokens

def get_vectorizer(max_features=5000, ngram_range=(1,2)):
    # Use TF-IDF with our clean_text preprocessor and tokenizer
    vect = TfidfVectorizer(preprocessor=clean_text, tokenizer=simple_tokenizer, max_features=max_features, ngram_range=ngram_range)
    return vect

def get_pipeline(model_name="MultinomialNB", vectorizer_params=None, clf_params=None):
    """
    Return sklearn Pipeline with TFIDF + estimator.
    """
    vectorizer_params = vectorizer_params or {}
    clf_params = clf_params or {}

    vect = get_vectorizer(**vectorizer_params)
    if model_name == "MultinomialNB":
        clf = MultinomialNB(**clf_params)
    elif model_name == "LogisticRegression":
        # solver 'liblinear' works well for small datasets and 3.8 compatibility.
        clf = LogisticRegression(solver='liblinear', max_iter=1000, **clf_params)
    else:
        raise ValueError("Unknown model: " + model_name)

    pipeline = Pipeline([
        ('tfidf', vect),
        ('clf', clf)
    ])
    return pipeline
