import streamlit as st
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet as wn
from collections import defaultdict
import re
import time
import pandas as pd

# -------------------------------------------------------------
# Page Configuration & Styling
# -------------------------------------------------------------
st.set_page_config(
    page_title="IR Assignment 1 - BITS WILP",
    layout="wide"
)

# -------------------------------------------------------------
# Resource Setup
# -------------------------------------------------------------
for resource in ["punkt", "punkt_tab", "stopwords", "wordnet", "omw-1.4"]:
    nltk.download(resource, quiet=True)

stemmer = PorterStemmer()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

# -------------------------------------------------------------
# Core Tree Data Structures
# -------------------------------------------------------------
class BSTNode:
    def __init__(self, key, postings=None):
        self.key = key
        self.postings = postings if postings is not None else set()
        self.left = None
        self.right = None

class BinarySearchTree:
    def __init__(self):
        self.root = None

    def insert(self, key, doc_id=None):
        if not self.root:
            self.root = BSTNode(key)
            if doc_id:
                self.root.postings.add(doc_id)
        else:
            self._insert(self.root, key, doc_id)

    def _insert(self, node, key, doc_id):
        if key == node.key:
            if doc_id:
                node.postings.add(doc_id)
        elif key < node.key:
            if node.left is None:
                node.left = BSTNode(key)
                if doc_id:
                    node.left.postings.add(doc_id)
            else:
                self._insert(node.left, key, doc_id)
        else:
            if node.right is None:
                node.right = BSTNode(key)
                if doc_id:
                    node.right.postings.add(doc_id)
            else:
                self._insert(node.right, key, doc_id)

    def search(self, key):
        node = self.root
        while node:
            if node.key == key:
                return True, node.postings
            node = node.left if key < node.key else node.right
        return False, set()

class BTreeNode:
    def __init__(self, t, leaf=False):
        self.t = t
        self.leaf = leaf
        self.keys = []
        self.postings = defaultdict(set)
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
            return True, node.postings[k]
        if node.leaf:
            return False, set()
        return self.search(k, node.children[i])

    def insert(self, k, doc_id=None):
        root = self.root
        if len(root.keys) == (2 * self.t) - 1:
            temp = BTreeNode(self.t, False)
            self.root = temp
            temp.children.insert(0, root)
            self.split_child(temp, 0)
            self.insert_non_full(temp, k, doc_id)
        else:
            self.insert_non_full(root, k, doc_id)

    def insert_non_full(self, x, k, doc_id):
        i = len(x.keys) - 1
        if x.leaf:
            if k in x.keys:
                if doc_id:
                    x.postings[k].add(doc_id)
                return
            x.keys.append(None)
            while i >= 0 and k < x.keys[i]:
                x.keys[i + 1] = x.keys[i]
                i -= 1
            x.keys[i + 1] = k
            if doc_id:
                x.postings[k].add(doc_id)
        else:
            while i >= 0 and k < x.keys[i]:
                i -= 1
            i += 1
            if len(x.children[i].keys) == (2 * self.t) - 1:
                self.split_child(x, i)
                if k > x.keys[i]:
                    i += 1
            self.insert_non_full(x.children[i], k, doc_id)

    def split_child(self, x, i):
        t = self.t
        y = x.children[i]
        z = BTreeNode(t, y.leaf)
        x.children.insert(i + 1, z)
        promoted_key = y.keys[t - 1]
        x.keys.insert(i, promoted_key)
        x.postings[promoted_key] = y.postings[promoted_key]
        
        z.keys = y.keys[t:(2 * t - 1)]
        for k in z.keys:
            z.postings[k] = y.postings[k]
        y.keys = y.keys[0:(t - 1)]
        
        if not y.leaf:
            z.children = y.children[t:(2 * t)]
            y.children = y.children[0:t]

# -------------------------------------------------------------
# Tolerant Retrieval Utilities
# -------------------------------------------------------------
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

def soundex(token):
    token = token.upper()
    if not token or not token.isalpha():
        return ""
    mapping = {
        'B': '1', 'F': '1', 'P': '1', 'V': '1',
        'C': '2', 'G': '2', 'J': '2', 'K': '2', 'Q': '2', 'S': '2', 'X': '2', 'Z': '2',
        'D': '3', 'T': '3',
        'L': '4',
        'M': '5', 'N': '5',
        'R': '6'
    }
    first_letter = token[0]
    encoded = [first_letter]
    prev_code = mapping.get(first_letter, "0")
    for char in token[1:]:
        code = mapping.get(char, "0")
        if code != "0":
            if code != prev_code:
                encoded.append(code)
            prev_code = code
        else:
            prev_code = "0"
    res = "".join(encoded[1:])
    return (first_letter + res + "000")[:4]

# -------------------------------------------------------------
# Document State Management
# -------------------------------------------------------------
if "documents" not in st.session_state:
    st.session_state.documents = {}

# -------------------------------------------------------------
# Header with BITS WILP Logo
# -------------------------------------------------------------
col_title, col_logo = st.columns([4, 1])

with col_title:
    st.title("Interactive Information Retrieval System")
    st.caption("BITS Pilani WILP · Information Retrieval (AIMLCZG537 / DSECLZG537) · Assignment 1")

with col_logo:
    st.image(
        "https://bits-pilani-wilp.ac.in/assets/images/logo.png",
        width=180
    )

tab1, tab2, tab3, tab4 = st.tabs([
    "1. Collection & Preprocessing",
    "2. Phrase Query & Indexes",
    "3. Dictionary Trees (BST vs B-Tree)",
    "4. Tolerant Retrieval"
])

# =============================================================
# TAB 1: Collection & Intermediate Preprocessing
# =============================================================
with tab1:
    st.subheader("1. Document Ingestion")
    uploaded_files = st.file_uploader(
        "Upload one or multiple plain text documents (.txt) to begin", 
        type=["txt"], 
        accept_multiple_files=True
    )
    if uploaded_files:
        new_docs = {}
        for uploaded_file in uploaded_files:
            new_docs[uploaded_file.name] = uploaded_file.read().decode("utf-8")
        st.session_state.documents = new_docs
        st.success(f"Successfully loaded {len(uploaded_files)} documents into memory!")

    if not st.session_state.documents:
        st.warning("⚠️ No documents uploaded yet. Please upload at least one .txt file above to proceed.")
    else:
        with st.expander("📂 View Uploaded Collection Content", expanded=True):
            num_cols = max(1, min(len(st.session_state.documents), 4))
            doc_cols = st.columns(num_cols)
            for idx, (doc_name, content) in enumerate(st.session_state.documents.items()):
                col_target = doc_cols[idx % num_cols]
                with col_target:
                    st.markdown(f"**{doc_name}**")
                    st.text_area(f"Raw content of {doc_name}", content, height=100, label_visibility="collapsed", disabled=True)

        st.markdown("---")
        st.subheader("2. Interactive Preprocessing Configuration")
        st.write("Configure each modular step of the pipeline:")
        
        cfg_col1, cfg_col2 = st.columns(2)
        with cfg_col1:
            st.markdown("**Core Text Operations**")
            opt_hyphens = st.checkbox("Handle Hyphens (replace '-' with space)", value=True)
            opt_lowercase = st.checkbox("Convert to Lowercase", value=True)
            opt_stopwords = st.checkbox("Remove Stop Words (English)", value=True)

        with cfg_col2:
            st.markdown("**Morphological Normalization**")
            opt_norm = st.radio(
                "Choose Normalization Technique:",
                [
                    "None (Raw tokens)",
                    "Stemming (Porter Stemmer)",
                    "Lemmatization (WordNet Lemmatizer)",
                    "Both (Lemmatization followed by Stemming)"
                ],
                index=1
            )

        st.markdown("---")
        st.subheader("3. Step-by-Step Intermediate Pipeline Outputs")
        doc_to_inspect = st.selectbox("Select document to inspect intermediate stages:", list(st.session_state.documents.keys()))
        raw_sample = st.session_state.documents[doc_to_inspect]

        # Stage 1: Hyphen Handling
        stage1_text = raw_sample.replace("-", " ") if opt_hyphens else raw_sample
        # Stage 2: Lowercasing
        stage2_text = stage1_text.lower() if opt_lowercase else stage1_text
        # Stage 3: Tokenization
        stage3_tokens = [w for w in word_tokenize(stage2_text) if w.isalnum()]
        # Stage 4: Stop Words
        stage4_tokens = [w for w in stage3_tokens if (w.lower() not in stop_words)] if opt_stopwords else stage3_tokens

        # Stage 5: Normalization
        if opt_norm == "Stemming (Porter Stemmer)":
            stage5_tokens = [stemmer.stem(w) for w in stage4_tokens]
        elif opt_norm == "Lemmatization (WordNet Lemmatizer)":
            stage5_tokens = [lemmatizer.lemmatize(w) for w in stage4_tokens]
        elif opt_norm == "Both (Lemmatization followed by Stemming)":
            stage5_tokens = [stemmer.stem(lemmatizer.lemmatize(w)) for w in stage4_tokens]
        else:
            stage5_tokens = stage4_tokens

        i_col1, i_col2 = st.columns(2)
        with i_col1:
            st.markdown("**Stage 0: Original Raw Text**")
            st.code(raw_sample, language="text")

            st.markdown(f"**Stage 1: Hyphen Handling {'(Active)' if opt_hyphens else '(Bypassed)'}**")
            st.code(stage1_text, language="text")

            st.markdown(f"**Stage 2: Lowercasing {'(Active)' if opt_lowercase else '(Bypassed)'}**")
            st.code(stage2_text, language="text")

        with i_col2:
            st.markdown("**Stage 3: Tokenization (Alphanumeric words)**")
            st.write(stage3_tokens)

            st.markdown(f"**Stage 4: Stop Words Filtered {'(Active)' if opt_stopwords else '(Bypassed)'}**")
            st.write(stage4_tokens)

            st.markdown(f"**Stage 5: Final Normalized Output ({opt_norm})**")
            st.success(f"Final Tokens: {stage5_tokens}")

        # -------------------------------------------------------------
        # 4. Multi-Metric Stemming vs Lemmatization Evaluation
        # -------------------------------------------------------------
        st.markdown("---")
        
        if st.button("Run Stemming vs Lemmatization Comparison"):
            raw_all_tokens = []
            stem_idx = defaultdict(set)
            lemma_idx = defaultdict(set)
            stem_vocab = set()
            lemma_vocab = set()

            for d_id, d_text in st.session_state.documents.items():
                clean_text = d_text.replace("-", " ").lower() if opt_hyphens else d_text.lower()
                toks = [w for w in word_tokenize(clean_text) if w.isalnum()]
                if opt_stopwords:
                    toks = [w for w in toks if w not in stop_words]
                
                raw_all_tokens.extend(toks)
                s_toks = [stemmer.stem(w) for w in toks]
                l_toks = [lemmatizer.lemmatize(w) for w in toks]

                stem_vocab.update(s_toks)
                lemma_vocab.update(l_toks)

                for t in set(s_toks):
                    stem_idx[t].add(d_id)
                for t in set(l_toks):
                    lemma_idx[t].add(d_id)

            raw_vocab_size = max(len(set(raw_all_tokens)), 1)
            stem_compression = round((1 - (len(stem_vocab) / raw_vocab_size)) * 100, 2)
            lemma_compression = round((1 - (len(lemma_vocab) / raw_vocab_size)) * 100, 2)

            valid_stem = sum(1 for w in stem_vocab if wn.synsets(w))
            valid_lemma = sum(1 for w in lemma_vocab if wn.synsets(w))
            stem_valid_rate = round((valid_stem / max(len(stem_vocab), 1)) * 100, 2)
            lemma_valid_rate = round((valid_lemma / max(len(lemma_vocab), 1)) * 100, 2)

            test_queries = []
            for d_id, d_text in st.session_state.documents.items():
                words = [w for w in word_tokenize(d_text.replace("-", " ").lower()) if w.isalpha() and w not in stop_words]
                if len(words) >= 2:
                    test_queries.append(f"{words[0]} {words[1]}")
                if len(test_queries) >= 3:
                    break
            
            while len(test_queries) < 3:
                fallbacks = ["fox jumps", "Stanford university", "lung cancer"]
                test_queries.append(fallbacks[len(test_queries)])

            def compute_metrics(retrieved, ground_truth):
                if not retrieved or not ground_truth:
                    return 0.0, 0.0, 0.0
                intersection = retrieved & ground_truth
                p = len(intersection) / len(retrieved)
                r = len(intersection) / len(ground_truth)
                f1 = (2 * p * r) / (p + r) if (p + r) > 0 else 0.0
                return round(p, 4), round(r, 4), round(f1, 4)

            comparison_rows = []
            stem_p_list, stem_r_list, stem_f1_list = [], [], []
            lemma_p_list, lemma_r_list, lemma_f1_list = [], [], []

            for q in test_queries:
                q_words = [w for w in word_tokenize(q.lower()) if w.isalnum()]
                
                gt_docs = set()
                for d_id, d_text in st.session_state.documents.items():
                    d_lower = d_text.replace("-", " ").lower()
                    if all(qw in d_lower for qw in q_words):
                        gt_docs.add(d_id)

                q_stem = [stemmer.stem(w) for w in q_words]
                stem_subsets = [stem_idx.get(w, set()) for w in q_stem]
                s_retrieved = set.intersection(*stem_subsets) if stem_subsets else set()

                q_lemma = [lemmatizer.lemmatize(w) for w in q_words]
                lemma_subsets = [lemma_idx.get(w, set()) for w in q_lemma]
                l_retrieved = set.intersection(*lemma_subsets) if lemma_subsets else set()

                sp, sr, sf1 = compute_metrics(s_retrieved, gt_docs)
                lp, lr, lf1 = compute_metrics(l_retrieved, gt_docs)

                stem_p_list.append(sp)
                stem_r_list.append(sr)
                stem_f1_list.append(sf1)
                lemma_p_list.append(lp)
                lemma_r_list.append(lr)
                lemma_f1_list.append(lf1)

                comparison_rows.append({
                    "query": q,
                    "stem_retrieved": list(s_retrieved),
                    "stem_P": sp,
                    "stem_R": sr,
                    "stem_F1": sf1,
                    "lemma_retrieved": list(l_retrieved),
                    "lemma_P": lp,
                    "lemma_R": lr,
                    "lemma_F1": lf1
                })

            eval_df = pd.DataFrame(comparison_rows)
            st.dataframe(eval_df, use_container_width=True)

            avg_stem_p = round(sum(stem_p_list) / max(len(stem_p_list), 1), 4)
            avg_stem_r = round(sum(stem_r_list) / max(len(stem_r_list), 1), 4)
            avg_stem_f1 = round(sum(stem_f1_list) / max(len(stem_f1_list), 1), 4)

            avg_lemma_p = round(sum(lemma_p_list) / max(len(lemma_p_list), 1), 4)
            avg_lemma_r = round(sum(lemma_r_list) / max(len(lemma_r_list), 1), 4)
            avg_lemma_f1 = round(sum(lemma_f1_list) / max(len(lemma_f1_list), 1), 4)

            st.markdown("**Summary (averaged across queries & corpus):**")
            summary_json = {
                "stemming": {
                    "avg_P": avg_stem_p,
                    "avg_R": avg_stem_r,
                    "avg_F1": avg_stem_f1,
                    "vocab_size": len(stem_vocab),
                    "compression_rate": f"{stem_compression}%",
                    "lexical_validity": f"{stem_valid_rate}%"
                },
                "lemmatization": {
                    "avg_P": avg_lemma_p,
                    "avg_R": avg_lemma_r,
                    "avg_F1": avg_lemma_f1,
                    "vocab_size": len(lemma_vocab),
                    "compression_rate": f"{lemma_compression}%",
                    "lexical_validity": f"{lemma_valid_rate}%"
                }
            }
            st.json(summary_json)

            # Strict comparison descriptions (no equal values called higher)
            if stem_compression > lemma_compression:
                comp_desc = f"Stemming achieved higher index compression ({stem_compression}% vs {lemma_compression}%), reducing vocabulary to {len(stem_vocab)} terms compared to {len(lemma_vocab)} for Lemmatization."
            elif lemma_compression > stem_compression:
                comp_desc = f"Lemmatization achieved higher index compression ({lemma_compression}% vs {stem_compression}%), reducing vocabulary to {len(lemma_vocab)} terms compared to {len(stem_vocab)} for Stemming."
            else:
                comp_desc = f"Both Stemming and Lemmatization achieved identical index compression ({stem_compression}%), each producing a vocabulary of {len(stem_vocab)} terms."

            if lemma_valid_rate > stem_valid_rate:
                val_desc = f"Lemmatization retained a higher proportion of authentic dictionary words ({lemma_valid_rate}% vs {stem_valid_rate}%)."
            elif stem_valid_rate > lemma_valid_rate:
                val_desc = f"Stemming retained a higher proportion of authentic dictionary words ({stem_valid_rate}% vs {lemma_valid_rate}%)."
            else:
                val_desc = f"Both Stemming and Lemmatization retained an identical proportion of authentic dictionary words ({lemma_valid_rate}%)."

            # Multi-Factor Composite Scoring Engine (50% Retrieval F1, 25% Compression, 25% Lexical Validity)
            stem_score = round((avg_stem_f1 * 0.5) + ((stem_compression / 100) * 0.25) + ((stem_valid_rate / 100) * 0.25), 4)
            lemma_score = round((avg_lemma_f1 * 0.5) + ((lemma_compression / 100) * 0.25) + ((lemma_valid_rate / 100) * 0.25), 4)

            st.markdown("### 🏆 Comprehensive Multi-Factor Evaluation")
            score_col1, score_col2 = st.columns(2)
            with score_col1:
                st.metric("Porter Stemming Composite Score", f"{stem_score}", delta=f"{round(stem_score - lemma_score, 4)} vs Lemma")
            with score_col2:
                st.metric("WordNet Lemmatization Composite Score", f"{lemma_score}", delta=f"{round(lemma_score - stem_score, 4)} vs Stem")

            if stem_score > lemma_score:
                st.success(f"""
                **Final Decision: Porter Stemming is more suitable for this corpus & query set.**
                
                **Data-Driven Justification:**
                - **Composite Score:** Stemming scored **{stem_score}** vs **{lemma_score}** for Lemmatization.
                - **Retrieval & Footprint:** Stemming achieved an F1 of {avg_stem_f1} and {stem_compression}% index compression.
                - **IR Trade-off:** The gains in match recall and dictionary footprint reduction ({comp_desc}) outweigh the difference in word validity ({val_desc}), making Stemming preferable for recall-oriented retrieval on this dataset.
                """)
            elif lemma_score > stem_score:
                st.success(f"""
                **Final Decision: WordNet Lemmatization is more suitable for this corpus & query set.**
                
                **Data-Driven Justification:**
                - **Composite Score:** Lemmatization scored **{lemma_score}** vs **{stem_score}** for Stemming.
                - **Semantic Precision & Validity:** {val_desc}
                - **Downstream Index Stability:** Retaining valid morphological roots prevents non-word stem collisions, ensuring phrase queries and edit distance spelling corrections remain accurate.
                """)
            else:
                st.info(f"""
                **Final Decision: Both techniques achieved an identical composite score ({stem_score}).**
                
                **Application-Specific Recommendation:**
                - Choose **Porter Stemming** if your deployment prioritizes **minimal memory/disk footprint** and **maximum recall**.
                - Choose **WordNet Lemmatization** if your deployment prioritizes **exact phrase precision** and **human-readable dictionary lookups**.
                """)

# =============================================================
# TAB 2: Inverted, Biword & Positional Indexing
# =============================================================
with tab2:
    st.subheader("Phrase Search: Biword vs. Positional Index")

    if not st.session_state.documents:
        st.warning("⚠️ Please upload documents in Tab 1 before building indexes or querying phrases.")
    else:
        def preprocess_document_text(text):
            t = text.replace("-", " ") if opt_hyphens else text
            t = t.lower() if opt_lowercase else t
            toks = [w for w in word_tokenize(t) if w.isalnum()]
            if opt_stopwords:
                toks = [w for w in toks if w.lower() not in stop_words]
            if opt_norm == "Stemming (Porter Stemmer)":
                toks = [stemmer.stem(w) for w in toks]
            elif opt_norm == "Lemmatization (WordNet Lemmatizer)":
                toks = [lemmatizer.lemmatize(w) for w in toks]
            elif opt_norm == "Both (Lemmatization followed by Stemming)":
                toks = [stemmer.stem(lemmatizer.lemmatize(w)) for w in toks]
            return toks

        inverted_index = defaultdict(set)
        biword_index = defaultdict(set)
        positional_index = defaultdict(lambda: defaultdict(list))

        for doc_id, text in st.session_state.documents.items():
            tokens = preprocess_document_text(text)
            for token in set(tokens):
                inverted_index[token].add(doc_id)
            for i in range(len(tokens) - 1):
                biword = f"{tokens[i]} {tokens[i+1]}"
                biword_index[biword].add(doc_id)
            for pos, token in enumerate(tokens):
                positional_index[token][doc_id].append(pos)

        with st.expander("🔍 View Active Inverted & Positional Postings"):
            icol1, icol2 = st.columns(2)
            with icol1:
                st.markdown("**Biword Index (Sample Entries)**")
                st.json({k: list(v) for k, v in list(biword_index.items())[:8]})
            with icol2:
                st.markdown("**Positional Index (Sample Entries)**")
                st.json({k: dict(v) for k, v in list(positional_index.items())[:5]})

        st.markdown("---")
        suggested_phrase = list(biword_index.keys())[0] if biword_index else ""
        phrase_query = st.text_input("Enter a phrase query to evaluate:", value=suggested_phrase)
        q_tokens = preprocess_document_text(phrase_query)

        if q_tokens:
            st.caption(f"Processed Query Tokens: `{q_tokens}`")
            if len(q_tokens) == 1:
                biword_hits = inverted_index.get(q_tokens[0], set())
                biword_pairs = [q_tokens[0]]
            else:
                biword_pairs = [f"{q_tokens[i]} {q_tokens[i+1]}" for i in range(len(q_tokens) - 1)]
                biword_subsets = [biword_index.get(f"{q_tokens[i]} {q_tokens[i+1]}", set()) for i in range(len(q_tokens) - 1)]
                biword_hits = set.intersection(*biword_subsets) if biword_subsets else set()

            def evaluate_positional(tokens, index):
                if not tokens or tokens[0] not in index:
                    return set(), {}
                candidates = set(index[tokens[0]].keys())
                for t in tokens[1:]:
                    candidates &= set(index[t].keys())
                valid_docs = set()
                matched_offsets = {}
                for doc in candidates:
                    for p in index[tokens[0]][doc]:
                        if all((p + offset) in index[term][doc] for offset, term in enumerate(tokens[1:], start=1)):
                            valid_docs.add(doc)
                            if doc not in matched_offsets:
                                matched_offsets[doc] = []
                            matched_offsets[doc].append(list(range(p, p + len(tokens))))
                            break
                return valid_docs, matched_offsets

            positional_hits, matched_offsets = evaluate_positional(q_tokens, positional_index)

            res_col1, res_col2 = st.columns(2)
            with res_col1:
                st.markdown("#### Biword Index Search Output")
                st.metric("Biword Index Matches", f"{len(biword_hits)} Documents")
                st.write(list(biword_hits))
            with res_col2:
                st.markdown("#### Positional Index Search Output")
                st.metric("Positional Index Matches", f"{len(positional_hits)} Documents")
                st.write(list(positional_hits))

            st.markdown("---")
            comp_col1, comp_col2 = st.columns(2)

            with comp_col1:
                st.markdown("### 1. Cases where Biword Index Gives False Positives")
                false_positives = biword_hits - positional_hits

                if len(q_tokens) >= 3:
                    if false_positives:
                        st.error(f"🚨 **False Positive Occurred!** Found in: `{sorted(list(false_positives))}`")
                        st.write(f"The query was decomposed into: `{biword_pairs}`")
                        st.write("All separate biword pairs matched in the document, but they were disjoint or in the wrong order.")
                    else:
                        st.success("✅ **No False Positives for this query:** Both indexes produced matching document sets.")
                else:
                    st.info("ℹ️ Enter a phrase of **3 or more words** to observe potential false positives.")

                st.write("""
                * **How it happens:** A query with 3 or more words ($A, B, C$) is split into independent pairs: `(A B)` AND `(B C)`.
                * **The failure case:** If a document contains the phrase `(B C)` in paragraph 1 and `(A B)` in paragraph 3, both individual pairs match. The biword intersection flags the document as a match even though the continuous phrase `(A B C)` never appears.
                """)

            with comp_col2:
                st.markdown("### 2. Why Positional Index Gives More Accurate Results")
                if positional_hits:
                    st.success(f"✅ Exact sequence verified in: `{sorted(list(positional_hits))}`")
                    for doc_name, positions in matched_offsets.items():
                        st.caption(f"**{doc_name}** matched at exact token positions: `{positions}`")
                else:
                    st.write("No exact contiguous sequence matched.")

                st.write("""
                * **Exact offset validation:** The positional index checks actual token coordinates within each document.
                * **Sequential order constraint:** It requires $\\text{position}(w_2) = \\text{position}(w_1) + 1$, $\\text{position}(w_3) = \\text{position}(w_2) + 1$, and so on.
                * **Precision guarantee:** By verifying both word order and exact adjacency, the positional index completely eliminates false positives caused by disjoint or inverted word pairs.
                """)

# =============================================================
# TAB 3: BST vs B-Tree Dictionary Search
# =============================================================
with tab3:
    st.subheader("Dictionary Search: Binary Search Tree vs. B-Tree")
    
    if not st.session_state.documents:
        st.warning("⚠️ Please upload documents in Tab 1 to build dictionary tree structures.")
    else:
        corpus_vocab = set()
        for text in st.session_state.documents.values():
            corpus_vocab.update(preprocess_document_text(text))

        bst = BinarySearchTree()
        btree = BTree(t=3)

        for doc_id, text in st.session_state.documents.items():
            doc_tokens = preprocess_document_text(text)
            for term in doc_tokens:
                bst.insert(term, doc_id)
                btree.insert(term, doc_id)

        vocab_sorted = sorted(list(corpus_vocab))
        st.write(f"Total Unique Vocabulary Terms (Under Active Preprocessing): **{len(corpus_vocab)}**")
        
        sample_terms_prompt = ", ".join(vocab_sorted[:3] + ["missingword"]) if len(vocab_sorted) >= 3 else ", ".join(vocab_sorted)
        terms_input = st.text_input("Enter comma-separated search terms for benchmarking:", value=sample_terms_prompt)

        if st.button("Run Performance Benchmark"):
            raw_test_terms = [t.strip() for t in terms_input.split(",") if t.strip()]
            test_terms = []
            for t in raw_test_terms:
                proc = preprocess_document_text(t)
                test_terms.append(proc[0] if proc else t.lower())

            iterations = 5000
            bst_search_times, btree_search_times = [], []
            bst_retrieval_times, btree_retrieval_times = [], []
            matches = []

            for term in test_terms:
                t0 = time.perf_counter()
                for _ in range(iterations):
                    found, _ = bst.search(term)
                t1 = time.perf_counter()
                bst_search_times.append(((t1 - t0) / iterations) * 1e6)

                t0 = time.perf_counter()
                for _ in range(iterations):
                    _, docs = bst.search(term)
                t1 = time.perf_counter()
                bst_retrieval_times.append(((t1 - t0) / iterations) * 1e6)

                t0 = time.perf_counter()
                for _ in range(iterations):
                    found, _ = btree.search(term)
                t1 = time.perf_counter()
                btree_search_times.append(((t1 - t0) / iterations) * 1e6)

                t0 = time.perf_counter()
                for _ in range(iterations):
                    _, docs = btree.search(term)
                t1 = time.perf_counter()
                btree_retrieval_times.append(((t1 - t0) / iterations) * 1e6)

                matches.append(found)

            bench_df = pd.DataFrame({
                "Query Term": test_terms,
                "In Corpus": matches,
                "BST Search (µs)": [round(x, 4) for x in bst_search_times],
                "B-Tree Search (µs)": [round(x, 4) for x in btree_search_times],
                "BST Retrieval (µs)": [round(x, 4) for x in bst_retrieval_times],
                "B-Tree Retrieval (µs)": [round(x, 4) for x in btree_retrieval_times],
            })
            st.table(bench_df)
            st.info("**Inference:** BST is fast in local memory because of simple pointer traversal. B-Tree remains balanced and scales better as the vocabulary becomes larger.")

# =============================================================
# TAB 4: Tolerant Retrieval (Wildcard, K-Gram, Edit Distance, Phonetic)
# =============================================================
with tab4:
    st.subheader("Tolerant Retrieval: Wildcard, K-Gram, Edit Distance & Phonetic Search")
    
    if not st.session_state.documents:
        st.warning("⚠️ Please upload documents in Tab 1 to generate vocabulary and run tolerant retrieval.")
    else:
        corpus_vocab = set()
        for text in st.session_state.documents.values():
            corpus_vocab.update(preprocess_document_text(text))
        vocab_list = sorted(list(corpus_vocab))
        alpha_vocab = [v for v in vocab_list if v.isalpha()]
        
        kgram_index = build_kgram_index(vocab_list, k=2)

        soundex_index = defaultdict(set)
        for term in alpha_vocab:
            code = soundex(term)
            if code:
                soundex_index[code].add(term)

        # Section A: Wildcard Queries & K-Gram Index
        st.markdown("### 1. Wildcard Queries via K-Gram (2-Gram) Index")
        sample_wildcard = ""
        for v in vocab_list:
            if len(v) >= 3 and v.isalpha():
                sample_wildcard = f"{v[0]}*{v[-1]}"
                break
        
        w_col1, w_col2 = st.columns([2, 1])
        with w_col1:
            wildcard = st.text_input("Enter wildcard pattern (e.g., 'te*gy', '*tech*', 'eng*'):", value=sample_wildcard)
        with w_col2:
            show_kgrams = st.checkbox("Show candidate 2-grams", value=True)

        if wildcard and "*" in wildcard:
            parts = wildcard.lower().split("*")
            grams = []
            for idx, part in enumerate(parts):
                if not part:
                    continue
                if idx == 0:
                    sub = f"${part}"
                elif idx == len(parts) - 1:
                    sub = f"{part}$"
                else:
                    sub = part
                if len(sub) >= 2:
                    grams.extend([sub[i:i+2] for i in range(len(sub) - 1)])

            if grams:
                candidates = set.intersection(*[kgram_index.get(g, set()) for g in grams])
                if show_kgrams:
                    st.caption(f"Decomposed Query 2-grams: `{grams}` | Pruned Candidates: `{len(candidates)}`")
            else:
                candidates = set(vocab_list)

            regex_pattern = f"^{wildcard.replace('*', '.*')}$"
            matches = [c for c in candidates if re.match(regex_pattern, c)]
            
            if matches:
                st.success(f"Matched Vocabulary Terms ({len(matches)}): `{sorted(matches)}`")
            else:
                st.write("No vocabulary words match the wildcard query.")

        with st.expander("🔍 View K-Gram Index (Sample 10 Keys)"):
            st.json({k: list(v) for k, v in list(kgram_index.items())[:10]})

        st.markdown("---")
        # Section B: Spelling & Edit Distance Correction
        st.markdown("### 2. Spelling Correction via Levenshtein Edit Distance")
        s_col1, s_col2 = st.columns(2)
        
        with s_col1:
            sample_typo = alpha_vocab[0][:-1] if alpha_vocab and len(alpha_vocab[0]) > 2 else "word"
            misspelled = st.text_input("Enter query term for spell correction:", value=sample_typo)
            max_dist = st.slider("Max Levenshtein Edit Distance Threshold", 1, 3, 2)
            
            if misspelled:
                suggestions = [
                    (term, edit_distance(misspelled.lower(), term))
                    for term in vocab_list
                    if edit_distance(misspelled.lower(), term) <= max_dist
                ]
                suggestions.sort(key=lambda x: x[1])
                if suggestions:
                    st.table(pd.DataFrame(suggestions, columns=["Suggested Correction", "Edit Distance"]))
                else:
                    st.write("No vocabulary terms found within edit distance threshold.")

        # Section C: Phonetic Correction (Soundex)
        with s_col2:
            st.markdown("### 3. Phonetic Correction (Soundex)")
            phonetic_sample = alpha_vocab[0] if alpha_vocab else "college"
            phonetic_query = st.text_input("Enter word for phonetic match (e.g. sounds similar):", value=phonetic_sample)
            
            if phonetic_query:
                query_soundex = soundex(phonetic_query)
                st.caption(f"Soundex Hash Code: `{query_soundex}`")
                phonetic_matches = sorted(list(soundex_index.get(query_soundex, set())))
                
                if phonetic_matches:
                    st.success(f"Phonetically Equivalent Words ({len(phonetic_matches)}):")
                    st.write(phonetic_matches)
                else:
                    st.write("No phonetically equivalent terms found in corpus vocabulary.")