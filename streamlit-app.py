import os
import re
import subprocess
import time

# -------------------------------------------------------------
# 1. Install Dependencies
# -------------------------------------------------------------
print("Installing dependencies...")
subprocess.run(["pip", "install", "-q", "streamlit", "nltk", "pandas"], check=True)

# Download cloudflared for tunneling
if not os.path.exists("cloudflared"):
    subprocess.run(["wget", "-q", "-nc", "https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64", "-O", "cloudflared"], check=True)
    subprocess.run(["chmod", "+x", "cloudflared"], check=True)

# -------------------------------------------------------------
# 2. Write app.py Directly via Python File Writer
# -------------------------------------------------------------
app_code = '''import streamlit as st
import re
import time
import pandas as pd
from collections import defaultdict
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer

for resource in ['stopwords', 'wordnet', 'omw-1.4']:
    try:
        nltk.data.find(f'corpora/{resource}')
    except LookupError:
        nltk.download(resource, quiet=True)

class BTreeNode:
    def __init__(self, t, leaf=False):
        self.t = t
        self.leaf = leaf
        self.keys = []
        self.children = []

class BTree:
    def __init__(self, t=3):
        self.root = BTreeNode(t, True)
        self.t = t

    def search(self, k, node=None):
        if node is None:
            node = self.root
        i = 0
        while i < len(node.keys) and k > node.keys[i]:
            i += 1
        if i < len(node.keys) and node.keys[i] == k:
            return True
        if node.leaf:
            return False
        return self.search(k, node.children[i])

    def insert(self, k):
        root = self.root
        if len(root.keys) == (2 * self.t) - 1:
            temp = BTreeNode(self.t, False)
            self.root = temp
            temp.children.insert(0, root)
            self.split_child(temp, 0)
            self.insert_non_full(temp, k)
        else:
            self.insert_non_full(root, k)

    def insert_non_full(self, x, k):
        i = len(x.keys) - 1
        if x.leaf:
            x.keys.append(None)
            while i >= 0 and k < x.keys[i]:
                x.keys[i + 1] = x.keys[i]
                i -= 1
            x.keys[i + 1] = k
        else:
            while i >= 0 and k < x.keys[i]:
                i -= 1
            i += 1
            if len(x.children[i].keys) == (2 * self.t) - 1:
                self.split_child(x, i)
                if k > x.keys[i]:
                    i += 1
            self.insert_non_full(x.children[i], k)

    def split_child(self, x, i):
        t = self.t
        y = x.children[i]
        z = BTreeNode(t, y.leaf)
        x.children.insert(i + 1, z)
        x.keys.insert(i, y.keys[t - 1])
        z.keys = y.keys[t:(2 * t - 1)]
        y.keys = y.keys[0:(t - 1)]
        if not y.leaf:
            z.children = y.children[t:(2 * t)]
            y.children = y.children[0:t]

class BSTNode:
    def __init__(self, key):
        self.key = key
        self.left = None
        self.right = None

class BST:
    def __init__(self):
        self.root = None

    def insert(self, key):
        if not self.root:
            self.root = BSTNode(key)
        else:
            self._insert(self.root, key)

    def _insert(self, node, key):
        if key < node.key:
            if node.left is None:
                node.left = BSTNode(key)
            else:
                self._insert(node.left, key)
        elif key > node.key:
            if node.right is None:
                node.right = BSTNode(key)
            else:
                self._insert(node.right, key)

    def search(self, key):
        return self._search(self.root, key)

    def _search(self, node, key):
        if node is None:
            return False
        if node.key == key:
            return True
        if key < node.key:
            return self._search(node.left, key)
        return self._search(node.right, key)

class TextPreprocessor:
    def __init__(self):
        self.stemmer = PorterStemmer()
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words = set(stopwords.words('english'))

    def process(self, text, lower=True, remove_hyphens=True, remove_stops=True, norm_type='stem'):
        if remove_hyphens:
            text = text.replace('-', ' ')
        if lower:
            text = text.lower()
        tokens = re.findall(r'\\b\\w+\\b', text)
        if remove_stops:
            tokens = [t for t in tokens if t not in self.stop_words]
        if norm_type == 'stem':
            tokens = [self.stemmer.stem(t) for t in tokens]
        elif norm_type == 'lemmatize':
            tokens = [self.lemmatizer.lemmatize(t) for t in tokens]
        return tokens

def edit_distance(s1, s2):
    m, n = len(s1), len(s2)
    dp = [[0] * (n + 1) for _ in range(m + 1)]
    for i in range(m + 1):
        dp[i][0] = i
    for j in range(n + 1):
        dp[0][j] = j
    for i in range(1, m + 1):
        for j in range(1, n + 1):
            if s1[i - 1] == s2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:
                dp[i][j] = 1 + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])
    return dp[m][n]

def build_kgram_index(vocab, k=2):
    kgram_idx = defaultdict(set)
    for term in vocab:
        padded = f"${term}$"
        for i in range(len(padded) - k + 1):
            gram = padded[i:i + k]
            kgram_idx[gram].add(term)
    return kgram_idx

st.set_page_config(page_title="Information Retrieval System", layout="wide")
st.title("Modular Information Retrieval System")

tabs = st.tabs([
    "1. Ingestion & Preprocessing", 
    "2. Inverted & Phrase Indexes", 
    "3. Tree Search Benchmarking", 
    "4. Tolerant Retrieval"
])

default_corpus = {
    "Doc_1": "The quick brown fox jumps over the lazy dog.",
    "Doc_2": "Never jump over a lazy dog, advised the brown fox.",
    "Doc_3": "Information retrieval and text mining are key fields of computer science.",
    "Doc_4": "Fast retrieval systems require optimized index data structures like B-Trees."
}

if 'corpus' not in st.session_state:
    st.session_state.corpus = default_corpus

preprocessor = TextPreprocessor()

with tabs[0]:
    st.header("Document Ingestion & Preprocessing Pipeline")
    uploaded_files = st.file_uploader("Upload Text Files (.txt)", type=["txt"], accept_multiple_files=True)
    if uploaded_files:
        st.session_state.corpus = {file.name: file.read().decode("utf-8") for file in uploaded_files}
        st.success(f"Loaded {len(uploaded_files)} documents.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Raw Corpus")
        for doc_id, content in st.session_state.corpus.items():
            st.text_area(f"[{doc_id}]", content, height=70, disabled=True)
            
    with col2:
        st.subheader("Preprocessing Options")
        p_lower = st.checkbox("Convert to Lowercase", value=True)
        p_hyphen = st.checkbox("Replace Hyphens with Spaces", value=True)
        p_stops = st.checkbox("Filter English Stop Words", value=True)
        p_norm = st.radio("Normalization Technique", ("Stemming (Porter)", "Lemmatization (WordNet)"))
        
        norm_key = 'stem' if 'Stemming' in p_norm else 'lemmatize'
        if st.button("Execute Preprocessing"):
            st.session_state.processed_corpus = {
                doc_id: preprocessor.process(text, p_lower, p_hyphen, p_stops, norm_key)
                for doc_id, text in st.session_state.corpus.items()
            }
            
        if 'processed_corpus' in st.session_state:
            st.subheader("Processed Tokens")
            for doc_id, toks in st.session_state.processed_corpus.items():
                st.write(f"**{doc_id}**: `{toks}`")

with tabs[1]:
    st.header("Biword & Positional Index Search")
    inverted_index = defaultdict(set)
    biword_index = defaultdict(set)
    positional_index = defaultdict(lambda: defaultdict(list))

    for doc_id, text in st.session_state.corpus.items():
        tokens = re.findall(r'\\b\\w+\\b', text.replace('-', ' ').lower())
        for token in set(tokens):
            inverted_index[token].add(doc_id)
        for i in range(len(tokens) - 1):
            biword_index[f"{tokens[i]} {tokens[i+1]}"].add(doc_id)
        for pos, token in enumerate(tokens):
            positional_index[token][doc_id].append(pos)

    col_idx1, col_idx2 = st.columns(2)
    with col_idx1:
        st.subheader("Biword Index Sample")
        st.json({k: list(v) for k, v in list(biword_index.items())[:8]})
    with col_idx2:
        st.subheader("Positional Index Sample")
        st.json({k: dict(v) for k, v in list(positional_index.items())[:5]})

    st.markdown("---")
    phrase_query = st.text_input("Enter Phrase Query (e.g., 'brown fox')", "brown fox")
    if phrase_query:
        p_tokens = re.findall(r'\\b\\w+\\b', phrase_query.lower())
        if len(p_tokens) == 1:
            biword_res = inverted_index.get(p_tokens[0], set())
        else:
            biword_matches = [biword_index.get(f"{p_tokens[i]} {p_tokens[i+1]}", set()) for i in range(len(p_tokens) - 1)]
            biword_res = set.intersection(*biword_matches) if biword_matches else set()

        def positional_search(tokens, index):
            if not tokens or tokens[0] not in index:
                return set()
            candidates = set(index[tokens[0]].keys())
            for t in tokens[1:]:
                candidates &= set(index[t].keys())
            matching_docs = set()
            for doc in candidates:
                for p in index[tokens[0]][doc]:
                    if all((p + offset) in index[term][doc] for offset, term in enumerate(tokens[1:], start=1)):
                        matching_docs.add(doc)
                        break
            return matching_docs

        pos_res = positional_search(p_tokens, positional_index)
        c1, c2 = st.columns(2)
        c1.write(f"**Biword Output:** `{list(biword_res)}`")
        c2.write(f"**Positional Output:** `{list(pos_res)}`")

with tabs[2]:
    st.header("Dictionary Search Comparison: BST vs B-Tree")
    raw_vocab = set()
    for text in st.session_state.corpus.values():
        raw_vocab.update(re.findall(r'\\b\\w+\\b', text.lower()))

    bst = BST()
    btree = BTree(t=3)
    for term in sorted(raw_vocab):
        bst.insert(term)
        btree.insert(term)

    st.write(f"**Vocabulary Size:** {len(raw_vocab)} terms")
    bench_queries = st.text_input("Enter comma-separated search terms", "fox, quick, data, unknown")
    
    if st.button("Run Search Benchmark"):
        terms = [t.strip().lower() for t in bench_queries.split(",") if t.strip()]
        bst_times, btree_times, found_status = [], [], []
        iterations = 5000

        for term in terms:
            t0 = time.perf_counter()
            for _ in range(iterations):
                res_bst = bst.search(term)
            bst_times.append(((time.perf_counter() - t0) / iterations) * 1e6)

            t0 = time.perf_counter()
            for _ in range(iterations):
                res_btree = btree.search(term)
            btree_times.append(((time.perf_counter() - t0) / iterations) * 1e6)
            found_status.append(res_bst)

        df = pd.DataFrame({
            "Query Term": terms,
            "Found": found_status,
            "BST Time (µs)": [round(t, 4) for t in bst_times],
            "B-Tree Time (µs)": [round(t, 4) for t in btree_times]
        })
        st.table(df)

with tabs[3]:
    st.header("Tolerant Retrieval: Wildcards & Spell Correction")
    vocab_list = sorted(list(raw_vocab))
    kgram_index = build_kgram_index(vocab_list, k=2)

    st.subheader("1. Wildcard Queries (2-Gram Index)")
    wildcard = st.text_input("Wildcard Search (e.g., 'f*x' or 're*')", "f*x")
    if wildcard and '*' in wildcard:
        parts = wildcard.lower().split('*')
        prefix_pattern = f"${parts[0]}" if parts[0] else ""
        suffix_pattern = f"{parts[1]}$" if parts[1] else ""
        target_grams = []
        if len(prefix_pattern) >= 2:
            target_grams.extend([prefix_pattern[i:i+2] for i in range(len(prefix_pattern)-1)])
        if len(suffix_pattern) >= 2:
            target_grams.extend([suffix_pattern[i:i+2] for i in range(len(suffix_pattern)-1)])
        
        candidates = set.intersection(*[kgram_index.get(g, set()) for g in target_grams]) if target_grams else set()
        matches = [c for c in candidates if re.match(f"^{wildcard.replace('*', '.*')}$", c)]
        st.write(f"**Matched Terms:** `{matches}`")

    st.markdown("---")
    st.subheader("2. Levenshtein Edit Distance Correction")
    query_word = st.text_input("Enter misspelled term", "quik")
    max_ed = st.slider("Maximum Edit Distance", 1, 3, 2)
    if query_word:
        ed_matches = sorted([(t, edit_distance(query_word.lower(), t)) for t in vocab_list if edit_distance(query_word.lower(), t) <= max_ed], key=lambda x: x[1])
        if ed_matches:
            st.table(pd.DataFrame(ed_matches, columns=["Suggested Term", "Edit Distance"]))
        else:
            st.info("No terms found within distance threshold.")
'''

with open("app.py", "w") as f:
    f.write(app_code)
print("app.py created successfully.")

# -------------------------------------------------------------
# 3. Launch Streamlit & Cloudflare Tunnel
# -------------------------------------------------------------
subprocess.run(["pkill", "-f", "streamlit"])
subprocess.run(["pkill", "-f", "cloudflared"])

print("Starting Streamlit...")
subprocess.Popen(["streamlit", "run", "app.py", "--server.port", "8501", "--server.headless", "true"])
time.sleep(3)

print("Starting Cloudflare Tunnel...")
tunnel_process = subprocess.Popen(
    ["./cloudflared", "tunnel", "--url", "http://localhost:8501"],
    stdout=subprocess.PIPE,
    stderr=subprocess.STDOUT,
    text=True
)

print("\nObtaining public URL...")
found_url = False
for _ in range(40):
    line = tunnel_process.stdout.readline()
    if "trycloudflare.com" in line:
        url_match = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', line)
        if url_match:
            print("\n" + "="*60)
            print(f"LIVE APP URL: {url_match.group(0)}")
            print("="*60 + "\n")
            found_url = True
            break
    time.sleep(0.5)

if not found_url:
    print("Cloudflare taking longer to start. Check process output or re-run the cell.")