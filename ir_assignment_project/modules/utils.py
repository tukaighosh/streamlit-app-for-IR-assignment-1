"""Utility helpers: loading uploaded files / the bundled sample dataset into
a {doc_id: text} corpus dict."""

import os


def load_corpus_from_uploads(uploaded_files):
    """
    uploaded_files: list of Streamlit UploadedFile objects (.txt) OR
                     a single .csv with columns [doc_id, text] (optional support).
    Returns: {doc_id: text}
    """
    corpus = {}
    for f in uploaded_files:
        name = f.name
        if name.lower().endswith(".csv"):
            import pandas as pd
            df = pd.read_csv(f)
            text_col = "text" if "text" in df.columns else df.columns[-1]
            id_col = "doc_id" if "doc_id" in df.columns else None
            for i, row in df.iterrows():
                doc_id = str(row[id_col]) if id_col else f"{name}_{i}"
                corpus[doc_id] = str(row[text_col])
        else:
            content = f.read()
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="ignore")
            doc_id = os.path.splitext(name)[0]
            corpus[doc_id] = content
    return corpus


def load_sample_corpus():
    """A small built-in sample dataset so the app works even before a user
    uploads anything (useful for the Virtual Lab demo)."""
    return {
        "doc1": "The quick brown fox jumps over the lazy dog. The fox was quick and brown.",
        "doc2": "A brown table stood in the room. Later, a fox jumped through the open window.",
        "doc3": "Stanford University released a new study on data science and machine learning.",
        "doc4": "The university of Stanford is well known, but this data science course is at MIT.",
        "doc5": "Deep learning and machine learning are subfields of artificial intelligence research.",
        "doc6": "Cancer biomarkers help researchers detect lung cancer at an early stage.",
        "doc7": "Researchers studying lung cancer biomarkers published new clinical trial results.",
        "doc8": "Photosynthesis is the process by which green plants convert sunlight into energy.",
        "doc9": "Information retrieval systems rank documents by relevance to a user query.",
        "doc10": "Search engines use inverted indexes to retrieve relevant documents quickly.",
        "doc11": "The dog was quick and brown, totally different from a brown fox that ran away.",
    }
