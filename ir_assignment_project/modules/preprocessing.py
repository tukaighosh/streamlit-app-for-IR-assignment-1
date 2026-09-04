"""
Text Preprocessing module for the IR Assignment.

Implements the pipeline discussed in Lecture 1 & Lecture 3:
    raw text -> lowercasing -> tokenization -> hyphen handling
    -> stop word removal -> stemming / lemmatization -> inverted index

Also implements a stemming-vs-lemmatization retrieval-quality comparison
using Precision / Recall / F1 (confusion-matrix framework from Lecture 2).
"""

import re
from collections import defaultdict

import nltk
from nltk.corpus import stopwords, wordnet
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag


def ensure_nltk_data():
    """Download required NLTK resources if not already present (safe to call every run)."""
    resources = [
        ("tokenizers/punkt", "punkt"),
        ("tokenizers/punkt_tab", "punkt_tab"),
        ("corpora/stopwords", "stopwords"),
        ("corpora/wordnet", "wordnet"),
        ("corpora/omw-1.4", "omw-1.4"),
        ("taggers/averaged_perceptron_tagger", "averaged_perceptron_tagger"),
        ("taggers/averaged_perceptron_tagger_eng", "averaged_perceptron_tagger_eng"),
    ]
    for path, pkg in resources:
        try:
            nltk.data.find(path)
        except LookupError:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass


STOPWORDS = None


def get_stopwords():
    global STOPWORDS
    if STOPWORDS is None:
        try:
            STOPWORDS = set(stopwords.words("english"))
        except LookupError:
            ensure_nltk_data()
            STOPWORDS = set(stopwords.words("english"))
    return STOPWORDS


# ---------------------------------------------------------------------------
# Individual pipeline steps (each returns intermediate output so the
# Streamlit app can display before/after at every stage)
# ---------------------------------------------------------------------------

def lowercase(text: str) -> str:
    return text.lower()


def handle_hyphens(text: str, mode: str = "split") -> str:
    """
    Hyphen handling is a design choice (Lecture 3), not a fixed rule.
    mode="split"  -> "state-of-the-art" -> "state of the art"  (splits on hyphen)
    mode="merge"  -> "state-of-the-art" -> "stateofoftheart"-like single token
                      (here: simply remove the hyphen -> "stateoftheart")
    mode="keep"   -> leave hyphenated words untouched
    """
    if mode == "split":
        return re.sub(r"(?<=\w)-(?=\w)", " ", text)
    elif mode == "merge":
        return re.sub(r"(?<=\w)-(?=\w)", "", text)
    return text


def tokenize(text: str):
    try:
        return word_tokenize(text)
    except LookupError:
        ensure_nltk_data()
        return word_tokenize(text)


def remove_punct_tokens(tokens):
    return [t for t in tokens if re.search(r"[A-Za-z0-9]", t)]


def remove_stopwords(tokens):
    sw = get_stopwords()
    return [t for t in tokens if t.lower() not in sw]


def stem_tokens(tokens):
    ps = PorterStemmer()
    return [ps.stem(t) for t in tokens]


def _wordnet_pos(tag):
    if tag.startswith("J"):
        return wordnet.ADJ
    if tag.startswith("V"):
        return wordnet.VERB
    if tag.startswith("N"):
        return wordnet.NOUN
    if tag.startswith("R"):
        return wordnet.ADV
    return wordnet.NOUN


def lemmatize_tokens(tokens):
    try:
        tagged = pos_tag(tokens)
    except LookupError:
        ensure_nltk_data()
        tagged = pos_tag(tokens)
    lemmatizer = WordNetLemmatizer()
    return [lemmatizer.lemmatize(tok, _wordnet_pos(tag)) for tok, tag in tagged]


# ---------------------------------------------------------------------------
# Full pipeline (returns every intermediate stage for display)
# ---------------------------------------------------------------------------

def run_pipeline(text: str, hyphen_mode="split", use_stemming=True, use_lemmatization=False):
    """
    Runs the complete preprocessing pipeline on one document/string.
    Returns a dict with every intermediate stage AND the final terms.
    """
    stages = {}
    stages["0_raw"] = text

    lowered = lowercase(text)
    stages["1_lowercased"] = lowered

    hyphen_handled = handle_hyphens(lowered, mode=hyphen_mode)
    stages["2_hyphen_handled"] = hyphen_handled

    tokens = tokenize(hyphen_handled)
    tokens = remove_punct_tokens(tokens)
    stages["3_tokenized"] = tokens

    no_stop = remove_stopwords(tokens)
    stages["4_stopwords_removed"] = no_stop

    final_terms = no_stop
    if use_stemming:
        final_terms = stem_tokens(final_terms)
        stages["5_stemmed"] = final_terms
    if use_lemmatization:
        final_terms = lemmatize_tokens(no_stop if not use_stemming else final_terms)
        stages["5_lemmatized"] = final_terms

    stages["final_terms"] = final_terms
    return stages


def preprocess_corpus(corpus: dict, hyphen_mode="split", use_stemming=True, use_lemmatization=False):
    """
    corpus: {doc_id: raw_text}
    Returns: {doc_id: [processed_terms]}
    """
    processed = {}
    for doc_id, text in corpus.items():
        stages = run_pipeline(text, hyphen_mode, use_stemming, use_lemmatization)
        processed[doc_id] = stages["final_terms"]
    return processed


# ---------------------------------------------------------------------------
# Inverted Index construction
# ---------------------------------------------------------------------------

def build_inverted_index(processed_corpus: dict):
    """
    processed_corpus: {doc_id: [terms]}
    Returns: {term: sorted list of doc_ids}
    """
    index = defaultdict(set)
    for doc_id, terms in processed_corpus.items():
        for t in terms:
            index[t].add(doc_id)
    return {term: sorted(list(docs)) for term, docs in sorted(index.items())}


# ---------------------------------------------------------------------------
# Stemming vs Lemmatization — retrieval quality comparison
# ---------------------------------------------------------------------------

def boolean_retrieve(query_terms, inverted_index):
    """Simple AND-based Boolean retrieval over an inverted index."""
    if not query_terms:
        return set()
    result = None
    for qt in query_terms:
        docs = set(inverted_index.get(qt, []))
        result = docs if result is None else (result & docs)
    return result or set()


def precision_recall_f1(retrieved: set, relevant: set):
    """Standard confusion-matrix based P/R/F1 (Lecture 2)."""
    if not retrieved and not relevant:
        return 1.0, 1.0, 1.0
    tp = len(retrieved & relevant)
    precision = tp / len(retrieved) if retrieved else 0.0
    recall = tp / len(relevant) if relevant else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0.0
    return precision, recall, f1


def compare_stemming_vs_lemmatization(corpus: dict, queries: list, relevant_sets: list, hyphen_mode="split"):
    """
    queries: list of raw query strings
    relevant_sets: list of sets of doc_ids considered relevant for each query
                   (ground truth you define manually for your small dataset)

    Returns a results table (list of dicts) + averaged P/R/F1 for both approaches.
    """
    stemmed_corpus = preprocess_corpus(corpus, hyphen_mode, use_stemming=True, use_lemmatization=False)
    lemma_corpus = preprocess_corpus(corpus, hyphen_mode, use_stemming=False, use_lemmatization=True)

    stemmed_index = build_inverted_index(stemmed_corpus)
    lemma_index = build_inverted_index(lemma_corpus)

    rows = []
    stem_metrics = []
    lemma_metrics = []

    for query, relevant in zip(queries, relevant_sets):
        q_stem = stem_tokens(remove_stopwords(tokenize(lowercase(query))))
        q_lemma = lemmatize_tokens(remove_stopwords(tokenize(lowercase(query))))

        stem_retrieved = boolean_retrieve(q_stem, stemmed_index)
        lemma_retrieved = boolean_retrieve(q_lemma, lemma_index)

        sp, sr, sf = precision_recall_f1(stem_retrieved, set(relevant))
        lp, lr, lf = precision_recall_f1(lemma_retrieved, set(relevant))

        stem_metrics.append((sp, sr, sf))
        lemma_metrics.append((lp, lr, lf))

        rows.append({
            "query": query,
            "stem_retrieved": sorted(stem_retrieved),
            "stem_P": round(sp, 2), "stem_R": round(sr, 2), "stem_F1": round(sf, 2),
            "lemma_retrieved": sorted(lemma_retrieved),
            "lemma_P": round(lp, 2), "lemma_R": round(lr, 2), "lemma_F1": round(lf, 2),
        })

    def avg(metrics, idx):
        return round(sum(m[idx] for m in metrics) / len(metrics), 3) if metrics else 0.0

    summary = {
        "stemming": {"avg_P": avg(stem_metrics, 0), "avg_R": avg(stem_metrics, 1), "avg_F1": avg(stem_metrics, 2),
                     "vocab_size": len(stemmed_index)},
        "lemmatization": {"avg_P": avg(lemma_metrics, 0), "avg_R": avg(lemma_metrics, 1), "avg_F1": avg(lemma_metrics, 2),
                           "vocab_size": len(lemma_index)},
    }
    return rows, summary
