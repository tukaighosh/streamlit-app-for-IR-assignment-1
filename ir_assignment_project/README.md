# IR Assignment 1 — End-to-End Information Retrieval System

Streamlit application covering: document upload, text preprocessing, phrase
query processing (biword vs positional index), dictionary search (BST vs
B-Tree), and tolerant retrieval (wildcard / edit distance / k-gram / soundex).

## Project structure

```
app.py                     # Main Streamlit app (run this)
modules/
  preprocessing.py         # Tokenization, stopwords, stemming/lemmatization, inverted index
  phrase_index.py          # Biword index + positional index + phrase query logic
  dict_trees.py            # BST + B-Tree implementations + timing experiment
  tolerant.py              # Wildcard/permuterm, edit distance, k-gram/Jaccard, Soundex
  utils.py                 # File upload helpers + bundled sample dataset
requirements.txt
```

## 1. Install dependencies

```bash
pip install -r requirements.txt
```

The app will automatically download the required NLTK data (punkt, stopwords,
wordnet, POS tagger) on first run — this needs an internet connection once.
If your environment has no internet access, pre-download manually:

```bash
python -c "import nltk; [nltk.download(p) for p in ['punkt','punkt_tab','stopwords','wordnet','omw-1.4','averaged_perceptron_tagger','averaged_perceptron_tagger_eng']]"
```

## 2. Run the app

```bash
streamlit run app.py
```

Then open the URL Streamlit prints (typically `http://localhost:8501`) in
your browser.

## 3. Using the app

Work through the tabs **in order**:

1. **Upload & View** — upload `.txt` files (or a `.csv` with `doc_id,text`
   columns), or click "Load bundled sample dataset" to try the app
   immediately without any files.
2. **Preprocessing** — choose hyphen-handling mode and stemming/lemmatization,
   inspect the pipeline stage-by-stage on a sample doc, build the inverted
   index for the whole corpus, and run the stemming-vs-lemmatization
   Precision/Recall/F1 comparison (define your own queries + relevant docs).
3. **Phrase Query** — build the biword and positional indexes, then run a
   phrase query to see both results side by side and any concrete false
   positives from the biword index.
4. **Dictionary Search** — build the BST and B-Tree from the vocabulary and
   run the timing experiment over a set of query terms.
5. **Tolerant Retrieval** — try wildcard queries, edit-distance spelling
   correction, k-gram/Jaccard similarity, and Soundex phonetic matching.
6. **Inference & Discussion** — write your conclusions for each rubric
   question and export them as Markdown for the report.

## 4. Notes for the report

- Take screenshots of each tab (with real output) for the "Demo evidence" and
  "Screenshots of Streamlit front end" submission requirements.
- The stemming-vs-lemmatization comparison, the BST-vs-B-Tree timing table,
  and the biword-vs-positional false-positive result are all designed to be
  copied straight into your report's experimental-results tables.
- Run the app in the BITS Virtual Lab (Prayog Shala) as required by the
  rubric, not just locally.
