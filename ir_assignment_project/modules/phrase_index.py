"""
Phrase Query Processing module (Lecture 3):
    - Biword (bigram) index          -> fast, but can give false positives
    - Positional index                -> slower/bigger, but exact-order accurate

Both are built from the SAME processed term lists used for the inverted
index, so results are directly comparable in the Streamlit UI.
"""

from collections import defaultdict


# ---------------------------------------------------------------------------
# Biword Index
# ---------------------------------------------------------------------------

def build_biword_index(processed_corpus: dict):
    """
    processed_corpus: {doc_id: [terms]}  (already preprocessed, in original order)
    Returns: {(term_i, term_i+1): sorted list of doc_ids}
    """
    index = defaultdict(set)
    for doc_id, terms in processed_corpus.items():
        for i in range(len(terms) - 1):
            bigram = (terms[i], terms[i + 1])
            index[bigram].add(doc_id)
    return {bg: sorted(docs) for bg, docs in index.items()}


def biword_phrase_query(query_terms, biword_index):
    """
    Splits the (already preprocessed) query into consecutive bigrams and
    ANDs their postings lists together. This is the source of false
    positives: it only confirms each PAIR co-occurs in the doc, not that
    the full phrase appears in that exact order end-to-end.
    """
    if len(query_terms) < 2:
        return set(), []
    bigrams = [(query_terms[i], query_terms[i + 1]) for i in range(len(query_terms) - 1)]
    result = None
    for bg in bigrams:
        docs = set(biword_index.get(bg, []))
        result = docs if result is None else (result & docs)
    return (result or set()), bigrams


# ---------------------------------------------------------------------------
# Positional Index
# ---------------------------------------------------------------------------

def build_positional_index(processed_corpus: dict):
    """
    Returns: {term: {doc_id: [positions]}}
    """
    index = defaultdict(lambda: defaultdict(list))
    for doc_id, terms in processed_corpus.items():
        for pos, term in enumerate(terms):
            index[term][doc_id].append(pos)
    return {term: dict(docs) for term, docs in index.items()}


def positional_phrase_query(query_terms, positional_index):
    """
    Merge-based algorithm (same spirit as Boolean AND merge, but comparing
    position offsets instead of doc IDs): a document matches only if there
    exists a starting position p such that term[0] is at p, term[1] is at
    p+1, term[2] is at p+2, ... i.e. the terms are truly consecutive.
    """
    if not query_terms:
        return set()

    # Candidate docs = intersection of docs containing every query term at all
    candidate_docs = None
    for t in query_terms:
        docs = set(positional_index.get(t, {}).keys())
        candidate_docs = docs if candidate_docs is None else (candidate_docs & docs)
    candidate_docs = candidate_docs or set()

    matched_docs = set()
    for doc_id in candidate_docs:
        first_term_positions = positional_index[query_terms[0]][doc_id]
        for start_pos in first_term_positions:
            if all(
                (start_pos + offset) in positional_index[term][doc_id]
                for offset, term in enumerate(query_terms)
            ):
                matched_docs.add(doc_id)
                break
    return matched_docs


def find_false_positives(query_terms, biword_index, positional_index):
    """
    Docs returned by the biword index but NOT by the positional index for
    the same phrase query -> concrete, dataset-specific false positives to
    show in the report/app (mirrors the Lecture 3 "quick brown fox" demo).
    """
    biword_result, bigrams = biword_phrase_query(query_terms, biword_index)
    positional_result = positional_phrase_query(query_terms, positional_index)
    false_positives = biword_result - positional_result
    return {
        "bigrams_used": bigrams,
        "biword_result": sorted(biword_result),
        "positional_result": sorted(positional_result),
        "false_positives": sorted(false_positives),
    }
