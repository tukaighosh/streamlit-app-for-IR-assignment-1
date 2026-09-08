# End-to-End Information Retrieval System (BITS WILP Assignment 1)

An interactive, modular Information Retrieval (IR) web application built using Python and Streamlit for **BITS Pilani WILP Information Retrieval (AIMLCZG537 / DSECLZG537)**. 

This application implements custom data structures and algorithms from scratch to demonstrate document preprocessing, multi-word phrase indexing, tree-based dictionary search benchmarking, tolerant retrieval, and automated rubric report inferences.

---

## 🚀 Quick Start & Installation

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system.

### 2. Install Dependencies
Install all required libraries specified in `requirements.txt`:
```bash
python -m pip install -r requirements.txt
```

### 3. Launch the Application
Run the Streamlit app locally with Python:
```cmd
python -m streamlit run app.py
```
*(Or `streamlit run app.py` if your Python Scripts folder is added to PATH)*.

---

## ✨ Features & Module Breakdown

1. **Tab 1: Document Collection & Preprocessing**:
   - Auto-loads standard sample documents (`sample_docs/`) on startup with option to upload custom `.txt` files.
   - Interactive preprocessing pipeline: hyphen handling, lowercasing, alphanumeric tokenization, and stop-word filtering.
   - Multi-metric Stemming (Porter) vs. Lemmatization (WordNet) evaluation with composite precision/recall scoring.

2. **Tab 2: Phrase Query Processing**:
   - Pure Python implementations of **Inverted Index**, **Biword Index**, and **Positional Index**.
   - Side-by-side search execution demonstrating why biword indexes yield false positives on 3+ word queries while positional indexes maintain exact positional coordinate precision.

3. **Tab 3: Dictionary Search (BST vs. B-Tree)**:
   - Pure Python implementations of a **Binary Search Tree (BST)** and a balanced **B-Tree** ($t=3$).
   - Benchmarking query search time and retrieval time ($\mu$s) across terms with tabular outputs.

4. **Tab 4: Tolerant Retrieval**:
   - **Wildcard Queries**: Resolved via **2-Gram (K-Gram) Index** with regex candidate pruning.
   - **Spelling Correction**: Driven by **Levenshtein Edit Distance** dynamic programming matrix.
   - **Phonetic Matching**: Soundex 4-character phonetic hashing for homophone discovery.

5. **Tab 5: Mandatory Assignment Inferences (Compulsory Rubric G)**:
   - Pre-populated answers and data-driven justifications for all 7 compulsory report questions (Best Preprocessing, Stemming vs. Lemmatization, Phrase Query Accuracy, Tree Speed, Tolerance, System Limitations, and Future Improvements).

---

## 🛠️ Bug Fixes & Improvements in Branch `fix/ir-app-assignment1`

- **B-Tree Internal Key Handling**: Resolved duplicate key creation and lost postings lists during B-Tree splits and updates.
- **BST Recursion Safety**: Converted tree insertion to iterative loop traversal to eliminate stack overflow on sorted inputs.
- **Positional Offset Recording**: Fixed premature loop termination (`break`) to capture all occurrence coordinate windows per document.
- **Accurate Evaluation Metrics**: Updated precision/recall ground-truth calculation to use tokenized word boundaries instead of raw string inclusion.
- **Spell Check Performance**: Optimized Levenshtein distance matrix computation to eliminate redundant evaluations per term.
- **Default Dataset Loader**: Added auto-loading of `sample_docs/` (`doc-1.txt` to `doc-4.txt`) for immediate usability.

---

## 📁 Repository Structure

```text
.
├── app.py                     # Main Streamlit application and core IR engine
├── requirements.txt           # Dependency requirements (streamlit, nltk, pandas)
├── sample_docs/               # Sample document collection (.txt)
├── test_assignment_fixes.py   # Unit test suite verifying core algorithms
└── README.md                  # Project setup and documentation
```

---

## 👤 Author & Credentials
- **Student Profile**: Gowtham-Dasari-15
- **Course**: BITS Pilani WILP - Information Retrieval Assignment 1