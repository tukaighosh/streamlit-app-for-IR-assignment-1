"""
Tolerant Retrieval module (Lecture 4 & 5):
    - Wildcard queries: prefix, suffix, middle (via Permuterm Index)
    - Spelling correction: Edit distance (Levenshtein)
    - Spelling correction: K-gram overlap (Jaccard similarity)
    - Phonetic correction: Soundex
"""

from collections import defaultdict


# ---------------------------------------------------------------------------
# Wildcard queries
# ---------------------------------------------------------------------------

def prefix_wildcard_search(prefix: str, dictionary_terms: list):
    """e.g. mon* -> all terms starting with 'mon'."""
    return sorted([t for t in dictionary_terms if t.startswith(prefix)])


def suffix_wildcard_search(suffix: str, dictionary_terms: list):
    """e.g. *tion -> reverse trick: reverse every term & the query, then prefix-match."""
    return sorted([t for t in dictionary_terms if t.endswith(suffix)])


def build_permuterm_index(dictionary_terms: list):
    """
    For each term, append '$' and generate all rotations.
    Returns: {rotation: original_term}
    """
    permuterm = {}
    for term in dictionary_terms:
        marked = term + "$"
        for i in range(len(marked)):
            rotation = marked[i:] + marked[:i]
            permuterm[rotation] = term
    return permuterm


def middle_wildcard_search(query: str, permuterm_index: dict):
    """
    Resolves a middle-wildcard query like 's*ndey' via permuterm rotation:
    1. Append '$', split on '*' -> rotate so the '*' portion moves to the end,
       turning it into a suffix (prefix-of-rotation) search.
    2. Find all permuterm dictionary entries whose rotation starts with that prefix.
    """
    if "*" not in query:
        return []
    before, after = query.split("*", 1)
    # rotate: we want pattern  after + "$" + before  as a PREFIX to search for
    search_prefix = after + "$" + before
    matches = set()
    for rotation, term in permuterm_index.items():
        if rotation.startswith(search_prefix):
            matches.add(term)
    return sorted(matches)


def wildcard_search(query: str, dictionary_terms: list, permuterm_index: dict = None):
    """Dispatch: decides prefix / suffix / middle based on '*' position."""
    if query.count("*") != 1:
        raise ValueError("Provide exactly one '*' in the wildcard query.")
    if query.startswith("*"):
        return suffix_wildcard_search(query[1:], dictionary_terms)
    elif query.endswith("*"):
        return prefix_wildcard_search(query[:-1], dictionary_terms)
    else:
        if permuterm_index is None:
            permuterm_index = build_permuterm_index(dictionary_terms)
        return middle_wildcard_search(query, permuterm_index)


# ---------------------------------------------------------------------------
# Edit Distance (Levenshtein) — DP matrix
# ---------------------------------------------------------------------------

def edit_distance(s1: str, s2: str):
    """Standard Levenshtein DP. Returns (distance, matrix) for display."""
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            cost = 0 if s1[i - 1] == s2[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,       # deletion
                dp[i][j - 1] + 1,       # insertion
                dp[i - 1][j - 1] + cost  # substitution / copy
            )
    return dp[m][n], dp


def closest_terms_by_edit_distance(query: str, dictionary_terms: list, top_k=5):
    scored = [(t, edit_distance(query, t)[0]) for t in dictionary_terms]
    scored.sort(key=lambda x: x[1])
    return scored[:top_k]


# ---------------------------------------------------------------------------
# K-gram overlap + Jaccard similarity
# ---------------------------------------------------------------------------

def get_kgrams(term: str, k=2):
    padded = f"${term}$"
    return set(padded[i:i + k] for i in range(len(padded) - k + 1))


def jaccard_similarity(set_a: set, set_b: set):
    if not set_a and not set_b:
        return 1.0
    union = set_a | set_b
    if not union:
        return 0.0
    return len(set_a & set_b) / len(union)


def closest_terms_by_kgram(query: str, dictionary_terms: list, k=2, top_k=5):
    q_grams = get_kgrams(query, k)
    scored = []
    for t in dictionary_terms:
        t_grams = get_kgrams(t, k)
        sim = jaccard_similarity(q_grams, t_grams)
        scored.append((t, round(sim, 3)))
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[:top_k]


def build_kgram_index(dictionary_terms: list, k=2):
    """k-gram -> set of terms containing it (candidate generation index)."""
    index = defaultdict(set)
    for term in dictionary_terms:
        for g in get_kgrams(term, k):
            index[g].add(term)
    return dict(index)


# ---------------------------------------------------------------------------
# Soundex (phonetic correction)
# ---------------------------------------------------------------------------

_SOUNDEX_CODES = {
    **{c: "1" for c in "BFPV"},
    **{c: "2" for c in "CGJKQSXZ"},
    **{c: "3" for c in "DT"},
    "L": "4",
    **{c: "5" for c in "MN"},
    "R": "6",
}
_IGNORED = set("AEIOUHWY")


def soundex(word: str) -> str:
    if not word:
        return ""
    word = word.upper()
    first_letter = word[0]
    codes = []
    prev_code = _SOUNDEX_CODES.get(first_letter, "")
    for ch in word[1:]:
        code = _SOUNDEX_CODES.get(ch, "")
        if ch in _IGNORED:
            prev_code = ""  # a vowel resets adjacency so repeated consonants across it are both kept
            continue
        if code and code != prev_code:
            codes.append(code)
        prev_code = code
    result = (first_letter + "".join(codes) + "000")[:4]
    return result


def phonetic_search(query: str, dictionary_terms: list):
    q_code = soundex(query)
    matches = [t for t in dictionary_terms if soundex(t) == q_code]
    return q_code, sorted(matches)
