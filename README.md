```text
$$\      $$\ $$\ $$\       $$\
$$ | $\  $$ |\__|$$ |      \__|
$$ |$$$\ $$ |$$\ $$ |  $$\ $$\  $$$$$$$\  $$$$$$\   $$$$$$\   $$$$$$\   $$$$$$$ |$$\  $$$$$$\
$$ $$ $$\$$ |$$ |$$ | $$  |$$ |$$  _____|$$  __$$\ $$  __$$\ $$  __$$\ $$  __$$ |$$ | \____$$\
$$$$  _$$$$ |$$ |$$$$$$  / $$ |\$$$$$$\  $$ /  $$ |$$$$$$$$ |$$$$$$$$ |$$ /  $$ |$$ | $$$$$$$ |
$$$  / \$$$ |$$ |$$  _$$<  $$ | \____$$\ $$ |  $$ |$$   ____|$$   ____|$$ |  $$ |$$ |$$  __$$ |
$$  /   \$$ |$$ |$$ | \$$\ $$ |$$$$$$$  |$$$$$$$  |\$$$$$$$\ \$$$$$$$\ \$$$$$$$ |$$ |\$$$$$$$ |
\__/     \__|\__|\__|  \__|\__|\_______/ $$  ____/  \_______| \_______| \_______|\__| \_______|
                                         $$ |
                                         $$ |
                                         \__|
```

# Wikispeedia

### Semantic Search for Wikipedia Using Transformer Embeddings

---

## Overview

Wikispeedia is a semantic search engine that indexes **1,000+ Wikipedia articles** and retrieves results based on **contextual meaning** rather than exact keyword matches.

Using transformer embeddings and PyTorch vector search, the engine maps text into a **384-dimensional semantic space**, allowing conceptually related content to be discovered even when the query terms never appear in the source text.

> Query: `feline`
> Result: Documents discussing `cats`

---

## Features

* Semantic search powered by transformer embeddings
* Indexes 1,000+ Wikipedia articles
* Fast vector similarity search using PyTorch
* Cosine similarity ranking across the entire corpus
* Pre-computed embeddings for immediate use

---

## How It Works

1. Wikipedia articles are cleaned and split into text chunks.
2. Each chunk is encoded using `all-MiniLM-L6-v2`.
3. Queries are embedded into the same vector space.
4. Cosine similarity ranks the most relevant chunks.
5. The highest-scoring results are returned to the user.

---

## Getting Started

### Prerequisites

```bash
pip install -r requirements.txt
```

### Run

Pre-computed embeddings and the processed dataset are included in the `data/` directory.

```bash
python main.py
```

---

## Project Structure

```text
├── data/
│   ├── dataset.pkl
│   └── finalembed.pt
├── onetimescripts/
│   ├── fetchdata.py
│   ├── cleandata.py
│   └── getembed.py
├── main.py
├── requirements.txt
└── README.md
```

---

### Built With

PyTorch • Hugging face
