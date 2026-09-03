# End-to-End Information Retrieval System

An interactive, modular Information Retrieval (IR) application built using Python and Streamlit. This application implements custom data structures and algorithms from scratch to demonstrate document preprocessing, multi-word phrase indexing, tree-based dictionary search benchmarking, and tolerant retrieval.

---

## Features

- **Document Ingestion & Preprocessing**:
  - Dynamic `.txt` document upload and real-time viewing.
  - Customizable text preprocessing: tokenization, lowercase conversion, hyphen normalization, and stop-word elimination.
  - Normalization engine comparing Porter Stemming vs. WordNet Lemmatization.
- **Phrase Query Processing**:
  - Custom implementations of standard **Inverted Index**, **Biword Index**, and **Positional Index**.
  - Side-by-side search execution demonstrating why biword indexes yield false positives on >2 word phrases while positional indexes maintain exact precision.
- **Dictionary Search (BST vs. B-Tree)**:
  - Custom, pure-Python implementations of a **Binary Search Tree (BST)** and a balanced **B-Tree** ($t=3$).
  - Microsecond-level lookup benchmarking across multiple queries with tabular performance metrics.
- **Tolerant Retrieval**:
  - **Wildcard Query Matching**: Permuted lookup handling leading/trailing wildcards (`*`) using a **2-Gram (K-Gram) Index** combined with regex verification.
  - **Spelling Correction**: Dynamic suggestion generation driven by dynamic-programming **Levenshtein Edit Distance**.

---

## Repository Structure

```text
.
├── app.py              # Main Streamlit application and core IR logic
├── requirements.txt    # Required external Python libraries
├── sample_docs/        # Directory containing sample corpus files (.txt)
└── README.md           # Project setup and documentation