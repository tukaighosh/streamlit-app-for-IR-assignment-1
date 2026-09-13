# End-to-End Information Retrieval System

An interactive, modular Information Retrieval (IR) application built using Python and Streamlit. This application implements custom data structures and algorithms from scratch to demonstrate document preprocessing, multi-word phrase indexing, tree-based dictionary search benchmarking, and tolerant retrieval.

---

## Features

- **Document Ingestion & Preprocessing**:
  - Dynamic `.txt` document upload and real-time viewing.
  - Customizable text preprocessing: tokenization, lowercase conversion, hyphen normalization, and stop-word elimination.
  - Normalization engine comparing Porter Stemming vs. WordNet Lemmatization.
- **Phrase Query Processing**:
  - Custom implementations of standard **Biword Index**, and **Positional Index**.
  - Side-by-side search execution demonstrating why biword indexes yield false positives on >2 word phrases while positional indexes maintain exact precision.
- **Dictionary Search (BST vs. B-Tree)**:
  - Custom, pure-Python implementations of a **Binary Search Tree (BST)** and a balanced **B-Tree** ($t=3$).
  - Microsecond-level lookup benchmarking across multiple queries with tabular performance metrics.
- **Tolerant Retrieval**:
  - **Wildcard Query Matching**: Permuted lookup handling leading/trailing wildcards (`*`) using a **2-Gram (K-Gram) Index** combined with regex verification.
  - **Spelling Correction**: Dynamic suggestion generation driven by dynamic-programming **Levenshtein Edit Distance**.
  - **Phonetic Correction**: Maps terms to 4-character phonetic codes based on how they sound, matching words that sound similar despite different spellin.

---

## Repository Structure

```text
.
├── app.py              # Main Streamlit application and core IR logic
├── sample_docs/        # Directory containing sample corpus files (.txt)
└── README.md           # Project setup and documentation
```

## Steps to run on BITS Virtual Environment

1.	Install required libraries : 
    - `pip install nltk pandas streamlit`

2.	Download the standalone cloudflared binary, set execution permissions, and move it to your system PATH(required to open cloudflared tunnel to overcome BITS environment limitation) :
    - `curl -L --output /usr/local/bin/cloudflared https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64 && chmod +x /usr/local/bin/cloudflared`

3.	Launch Streamlit app and receive localhost, network and external urls to connect the app.
    - `streamlit run app.py --server.address 0.0.0.0 --server.port 8501`

But, as BITS virtual environment doesn't have any browser to render UI, we need to go for step 4 to open cloudflare tunnel

4.	Start Cloudflare Tunnel and get the public URL immediately :
    - `cloudflared tunnel --url http://127.0.0.1:8501`

5. Use the cloudflare url to access to streamlit app from anywhere.
