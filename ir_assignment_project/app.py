"""
IR Assignment 1 — End-to-End Information Retrieval System
Streamlit front end covering: preprocessing, phrase query (biword vs
positional), dictionary search (BST vs B-Tree), and tolerant retrieval
(wildcard / edit distance / k-gram / soundex).

Run with:
    streamlit run app.py
"""

import time

import pandas as pd
import streamlit as st

from modules import preprocessing as prep
from modules import phrase_index as phr
from modules import dict_trees as trees
from modules import tolerant as tol
from modules.utils import load_corpus_from_uploads, load_sample_corpus

st.set_page_config(page_title="IR Assignment 1", layout="wide")
prep.ensure_nltk_data()

# ---------------------------------------------------------------------------
# Session state initialization
# ---------------------------------------------------------------------------

if "corpus" not in st.session_state:
    st.session_state.corpus = {}
if "hyphen_mode" not in st.session_state:
    st.session_state.hyphen_mode = "split"
if "use_stemming" not in st.session_state:
    st.session_state.use_stemming = True
if "use_lemmatization" not in st.session_state:
    st.session_state.use_lemmatization = False

st.title("📚 End-to-End Information Retrieval System")
st.caption("BITS WILP — Information Retrieval (AIMLCZG537 / DSECLZG537) — Assignment 1")

tabs = st.tabs([
    "1️⃣ Upload & View",
    "2️⃣ Preprocessing",
    "3️⃣ Phrase Query",
    "4️⃣ Dictionary Search (BST vs B-Tree)",
    "5️⃣ Tolerant Retrieval",
    "6️⃣ Inference & Discussion",
])

# ===========================================================================
# TAB 1 — Upload & View
# ===========================================================================
with tabs[0]:
    st.header("Upload a Document Collection")
    st.write("Upload one or more `.txt` files (or a `.csv` with `doc_id,text` columns), "
             "or use the bundled sample dataset to try the app immediately.")

    col1, col2 = st.columns([2, 1])
    with col1:
        uploaded = st.file_uploader(
            "Upload text documents", type=["txt", "csv"], accept_multiple_files=True
        )
    with col2:
        st.write("")
        st.write("")
        if st.button("Load bundled sample dataset instead"):
            st.session_state.corpus = load_sample_corpus()
            st.success(f"Loaded {len(st.session_state.corpus)} sample documents.")

    if uploaded:
        st.session_state.corpus = load_corpus_from_uploads(uploaded)
        st.success(f"Loaded {len(st.session_state.corpus)} document(s).")

    if st.session_state.corpus:
        st.subheader("Uploaded Documents")
        for doc_id, text in st.session_state.corpus.items():
            with st.expander(f"📄 {doc_id}"):
                st.write(text)
    else:
        st.info("No documents loaded yet. Upload files or load the sample dataset above.")

# ===========================================================================
# TAB 2 — Preprocessing
# ===========================================================================
with tabs[1]:
    st.header("Text Preprocessing Pipeline")

    if not st.session_state.corpus:
        st.warning("Please upload documents (or load the sample dataset) in Tab 1 first.")
    else:
        c1, c2, c3 = st.columns(3)
        with c1:
            hyphen_mode = st.selectbox(
                "Hyphen handling", ["split", "merge", "keep"],
                index=["split", "merge", "keep"].index(st.session_state.hyphen_mode),
                help="split: 'state-of-the-art' -> 'state of the art'. "
                     "merge: removes hyphen -> 'stateoftheart'. keep: leave untouched."
            )
        with c2:
            use_stemming = st.checkbox("Apply Stemming (Porter)", value=st.session_state.use_stemming)
        with c3:
            use_lemmatization = st.checkbox("Apply Lemmatization (WordNet)", value=st.session_state.use_lemmatization)

        st.session_state.hyphen_mode = hyphen_mode
        st.session_state.use_stemming = use_stemming
        st.session_state.use_lemmatization = use_lemmatization

        st.subheader("Step-by-step effect on a sample document")
        sample_doc_id = st.selectbox("Pick a document to inspect", list(st.session_state.corpus.keys()))
        stages = prep.run_pipeline(
            st.session_state.corpus[sample_doc_id], hyphen_mode, use_stemming, use_lemmatization
        )
        for stage_name, value in stages.items():
            st.markdown(f"**{stage_name}**")
            st.code(value if isinstance(value, str) else " | ".join(map(str, value)))

        st.subheader("Build Inverted Index (whole corpus)")
        if st.button("Build Inverted Index"):
            processed = prep.preprocess_corpus(st.session_state.corpus, hyphen_mode, use_stemming, use_lemmatization)
            index = prep.build_inverted_index(processed)
            st.session_state.processed_corpus = processed
            st.session_state.inverted_index = index
            st.success(f"Inverted index built with {len(index)} unique terms.")
            df = pd.DataFrame(
                [{"term": t, "postings": ", ".join(docs)} for t, docs in index.items()]
            )
            st.dataframe(df, use_container_width=True, height=350)

        st.subheader("Compare Stemming vs Lemmatization (Retrieval Quality)")
        st.write("Define 2–3 sample queries and the doc_ids you consider relevant (ground truth), "
                 "then compare Precision / Recall / F1 for stemmed vs lemmatized retrieval.")
        default_queries = "fox jumps\nStanford university\nlung cancer"
        queries_text = st.text_area("Queries (one per line)", value=default_queries)
        default_relevant = "doc1,doc2\ndoc3,doc4\ndoc6,doc7"
        relevant_text = st.text_area(
            "Relevant doc_ids per query (comma-separated, one line per query, same order)",
            value=default_relevant
        )
        if st.button("Run Stemming vs Lemmatization Comparison"):
            queries = [q.strip() for q in queries_text.splitlines() if q.strip()]
            relevant_sets = [
                [d.strip() for d in line.split(",") if d.strip()]
                for line in relevant_text.splitlines() if line.strip()
            ]
            if len(queries) != len(relevant_sets):
                st.error("Number of queries and relevant-doc lines must match.")
            else:
                rows, summary = prep.compare_stemming_vs_lemmatization(
                    st.session_state.corpus, queries, relevant_sets, hyphen_mode
                )
                st.dataframe(pd.DataFrame(rows), use_container_width=True)
                st.write("**Summary (averaged across queries):**")
                st.json(summary)
                better = "Stemming" if summary["stemming"]["avg_F1"] >= summary["lemmatization"]["avg_F1"] else "Lemmatization"
                st.success(f"On this dataset & query set, **{better}** achieved the higher average F1 score.")

# ===========================================================================
# TAB 3 — Phrase Query (Biword vs Positional)
# ===========================================================================
with tabs[2]:
    st.header("Phrase Query Processing — Biword Index vs Positional Index")

    if "processed_corpus" not in st.session_state:
        st.warning("Build the inverted index in Tab 2 first (uses the same preprocessed terms).")
    else:
        processed = st.session_state.processed_corpus

        if st.button("Build Biword & Positional Indexes"):
            st.session_state.biword_index = phr.build_biword_index(processed)
            st.session_state.positional_index = phr.build_positional_index(processed)
            st.success(
                f"Biword index: {len(st.session_state.biword_index)} bigrams | "
                f"Positional index: {len(st.session_state.positional_index)} terms"
            )

        if "biword_index" in st.session_state:
            st.subheader("Run a Phrase Query")
            phrase_query = st.text_input("Enter a phrase query (e.g. 'quick brown fox')", value="quick brown fox")
            if st.button("Search Phrase"):
                q_tokens = prep.remove_stopwords(prep.tokenize(prep.lowercase(phrase_query)))
                if st.session_state.use_stemming:
                    q_tokens = prep.stem_tokens(q_tokens)
                elif st.session_state.use_lemmatization:
                    q_tokens = prep.lemmatize_tokens(q_tokens)

                result = phr.find_false_positives(
                    q_tokens, st.session_state.biword_index, st.session_state.positional_index
                )
                c1, c2 = st.columns(2)
                with c1:
                    st.markdown("### 🔹 Biword Index Result")
                    st.write("Bigrams checked:", result["bigrams_used"])
                    st.write("Matching docs:", result["biword_result"])
                with c2:
                    st.markdown("### 🔹 Positional Index Result")
                    st.write("Matching docs:", result["positional_result"])

                if result["false_positives"]:
                    st.error(
                        f"⚠️ False positive(s) from the biword index: **{result['false_positives']}** — "
                        "these documents contain each consecutive word-pair somewhere, but NOT the exact "
                        "phrase in order. The positional index correctly excludes them."
                    )
                else:
                    st.info("No false positives found for this query on the current dataset — "
                            "try a query where the bigram pairs are shared across non-adjacent occurrences.")

            st.subheader("Inspect raw index structures")
            with st.expander("Biword Index (sample)"):
                sample = dict(list(st.session_state.biword_index.items())[:25])
                st.json({f"{k[0]} {k[1]}": v for k, v in sample.items()})
            with st.expander("Positional Index (sample)"):
                sample = dict(list(st.session_state.positional_index.items())[:25])
                st.json(sample)

# ===========================================================================
# TAB 4 — Dictionary Search: BST vs B-Tree
# ===========================================================================
with tabs[3]:
    st.header("Dictionary Search — Binary Search Tree vs B-Tree")

    if "inverted_index" not in st.session_state:
        st.warning("Build the inverted index in Tab 2 first.")
    else:
        dictionary = st.session_state.inverted_index
        st.write(f"Dictionary size: **{len(dictionary)}** unique terms.")

        t_order = st.slider("B-Tree minimum degree (t)", min_value=2, max_value=6, value=3)
        all_terms = list(dictionary.keys())

        st.subheader("Run Timing Experiment")
        default_n = min(10, len(all_terms))
        num_queries = st.slider("Number of random query terms", min_value=3,
                                 max_value=max(3, len(all_terms)), value=default_n)
        include_missing = st.checkbox("Include a few non-existent terms too (worst case)", value=True)

        if st.button("Run BST vs B-Tree Experiment"):
            import random
            sample_terms = random.sample(all_terms, min(num_queries, len(all_terms)))
            if include_missing:
                sample_terms += ["zzzzz_notfound", "xyz_missing"]

            rows, summary, bst, btree = trees.run_timing_experiment(dictionary, sample_terms, t=t_order)
            st.session_state.bst = bst
            st.session_state.btree = btree

            st.dataframe(pd.DataFrame(rows), use_container_width=True)
            st.write("**Summary:**")
            st.json(summary)

            chart_df = pd.DataFrame({
                "BST (µs)": [r["bst_time_us"] for r in rows],
                "B-Tree (µs)": [r["btree_time_us"] for r in rows],
            }, index=[r["term"] for r in rows])
            st.bar_chart(chart_df)

            st.info(
                f"On this run, **{summary['faster_structure']}** was faster on average. "
                "Note: on small in-memory dictionaries, BST can be as fast as or faster than a B-Tree — "
                "the B-Tree's real advantage (guaranteed O(log M), resistance to skew, fewer disk seeks) "
                "shows up at much larger, disk-resident vocabularies."
            )

        if "bst" in st.session_state:
            st.subheader("Try a single lookup")
            lookup_term = st.text_input("Term to search", value=all_terms[0] if all_terms else "")
            if st.button("Search single term"):
                bst_res, bst_visited = st.session_state.bst.search(lookup_term)
                btree_res, btree_visited = st.session_state.btree.search(lookup_term)
                st.write(f"**BST** — found: {bst_res is not None}, nodes visited: {bst_visited}, postings: {bst_res}")
                st.write(f"**B-Tree** — found: {btree_res is not None}, nodes visited: {btree_visited}, postings: {btree_res}")

# ===========================================================================
# TAB 5 — Tolerant Retrieval
# ===========================================================================
with tabs[4]:
    st.header("Tolerant Retrieval")

    if "inverted_index" not in st.session_state:
        st.warning("Build the inverted index in Tab 2 first.")
    else:
        all_terms = list(st.session_state.inverted_index.keys())
        technique = st.radio(
            "Choose a tolerant retrieval technique to demonstrate",
            ["Wildcard Query", "Edit Distance (Spelling Correction)", "K-gram / Jaccard Similarity", "Soundex (Phonetic)"],
            horizontal=True,
        )

        if technique == "Wildcard Query":
            st.write("Enter a query with exactly one `*` — prefix (`mon*`), suffix (`*tion`), or middle (`s*ndey`).")
            wq = st.text_input("Wildcard query", value=(all_terms[0][:2] + "*") if all_terms else "te*")
            if st.button("Run Wildcard Search"):
                try:
                    if "*" in wq and not wq.startswith("*") and not wq.endswith("*"):
                        permuterm = tol.build_permuterm_index(all_terms)
                        matches = tol.wildcard_search(wq, all_terms, permuterm)
                        st.caption("Resolved via Permuterm Index rotation (middle wildcard).")
                    else:
                        matches = tol.wildcard_search(wq, all_terms)
                    st.write(f"Matched **{len(matches)}** term(s):", matches)
                    if matches:
                        union_docs = set()
                        for m in matches:
                            union_docs |= set(st.session_state.inverted_index.get(m, []))
                        st.write("Documents containing any matched term:", sorted(union_docs))
                except ValueError as e:
                    st.error(str(e))

        elif technique == "Edit Distance (Spelling Correction)":
            st.write("Type a (possibly misspelled) term; the app finds the closest dictionary terms by Levenshtein distance.")
            typo = st.text_input("Query term", value="korrection")
            top_k = st.slider("Top-K closest terms", 1, 10, 5)
            if st.button("Find Closest Terms (Edit Distance)"):
                results = tol.closest_terms_by_edit_distance(typo, all_terms, top_k)
                st.dataframe(pd.DataFrame(results, columns=["term", "edit_distance"]), use_container_width=True)
                if results:
                    best_term = results[0][0]
                    dist, matrix = tol.edit_distance(typo, best_term)
                    st.write(f"DP matrix for `{typo}` → `{best_term}` (distance = {dist}):")
                    st.dataframe(pd.DataFrame(matrix, index=["-"] + list(typo), columns=["-"] + list(best_term)))
                    st.write("Documents for corrected term:", st.session_state.inverted_index.get(best_term, []))

        elif technique == "K-gram / Jaccard Similarity":
            st.write("Lightweight alternative to edit distance: compares k-gram sets via the Jaccard coefficient.")
            typo = st.text_input("Query term", value="korrection", key="kgram_query")
            k = st.slider("k (gram size)", 2, 3, 2)
            top_k = st.slider("Top-K closest terms", 1, 10, 5, key="kgram_topk")
            if st.button("Find Closest Terms (K-gram/Jaccard)"):
                results = tol.closest_terms_by_kgram(typo, all_terms, k=k, top_k=top_k)
                st.dataframe(pd.DataFrame(results, columns=["term", "jaccard_similarity"]), use_container_width=True)
                st.caption(f"Query k-grams: {sorted(tol.get_kgrams(typo, k))}")

        elif technique == "Soundex (Phonetic)":
            st.write("Finds dictionary terms that *sound* similar via the Soundex phonetic code.")
            word = st.text_input("Query word", value="smith" if "smith" not in all_terms else all_terms[0])
            if st.button("Run Soundex Search"):
                code, matches = tol.phonetic_search(word, all_terms)
                st.write(f"Soundex code for `{word}`: **{code}**")
                st.write("Phonetically matching dictionary terms:", matches)

# ===========================================================================
# TAB 6 — Inference & Discussion
# ===========================================================================
with tabs[5]:
    st.header("Inference & Discussion")
    st.write("Fill this section in after running the experiments above — the rubric requires "
             "answers to all 7 questions below, grounded in your own results (not generic answers).")

    q_and_a = [
        "1. Which preprocessing technique improved retrieval quality?",
        "2. Was stemming or lemmatization better for your dataset?",
        "3. Which phrase query index was more accurate?",
        "4. Which tree structure was faster?",
        "5. How tolerant was your retrieval model?",
        "6. What are the limitations of your system?",
        "7. How can the system be improved?",
    ]
    answers = {}
    for q in q_and_a:
        answers[q] = st.text_area(q, key=q, height=80)

    if st.button("Export Inference Answers as Markdown"):
        md = "\n\n".join(f"**{q}**\n\n{answers[q] or '_(not yet answered)_'}" for q in q_and_a)
        st.download_button("Download inference.md", data=md, file_name="inference.md", mime="text/markdown")
        st.markdown(md)

st.sidebar.title("ℹ️ About")
st.sidebar.info(
    "This app was built for BITS WILP Information Retrieval Assignment 1. "
    "Navigate through the tabs in order — each stage builds on the previous one "
    "(preprocessing → phrase query / dictionary search / tolerant retrieval)."
)
if st.session_state.corpus:
    st.sidebar.success(f"Corpus loaded: {len(st.session_state.corpus)} documents")
if "inverted_index" in st.session_state:
    st.sidebar.success(f"Vocabulary size: {len(st.session_state.inverted_index)} terms")
