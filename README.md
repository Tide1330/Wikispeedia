$$\      $$\ $$\ $$\       $$\                                               $$\ $$\           
$$ | $\  $$ |\__|$$ |      \__|                                              $$ |\__|          
$$ |$$$\ $$ |$$\ $$ |  $$\ $$\  $$$$$$$\  $$$$$$\   $$$$$$\   $$$$$$\   $$$$$$$ |$$\  $$$$$$\  
$$ $$ $$\$$ |$$ |$$ | $$  |$$ |$$  _____|$$  __$$\ $$  __$$\ $$  __$$\ $$  __$$ |$$ | \____$$\ 
$$$$  _$$$$ |$$ |$$$$$$  / $$ |\$$$$$$\  $$ /  $$ |$$$$$$$$ |$$$$$$$$ |$$ /  $$ |$$ | $$$$$$$ |
$$$  / \$$$ |$$ |$$  _$$<  $$ | \____$$\ $$ |  $$ |$$   ____|$$   ____|$$ |  $$ |$$ |$$  __$$ |
$$  /   \$$ |$$ |$$ | \$$\ $$ |$$$$$$$  |$$$$$$$  |\$$$$$$$\ \$$$$$$$\ \$$$$$$$ |$$ |\$$$$$$$ |
\__/     \__|\__|\__|  \__|\__|\_______/ $$  ____/  \_______| \_______| \_______|\__| \_______|
                                         $$ |                                                  
                                         $$ |                                                  
                                         \__|                                                  

An AI-powered semantic search engine that indexes 1,000+ Wikipedia articles and retrieves results based on meaning rather than exact keyword matches.

Using transformer embeddings and PyTorch vector search, Wikispeedia maps text into a 384-dimensional semantic space, allowing it to understand conceptual relationships between terms. For example, a search for **"feline"** can return articles discussing **"cats"** even when the word *feline* never appears in the text.

## Features

* Semantic search powered by transformer embeddings
* Indexes 1,000 Wikipedia articles
* Fast vector similarity search using PyTorch
* Cosine similarity ranking across the entire corpus
* Pre-computed embeddings for instant querying

## How It Works

1. Wikipedia articles are cleaned and split into text chunks.
2. Each chunk is converted into a 384-dimensional embedding using `all-MiniLM-L6-v2`.
3. Queries are embedded into the same vector space.
4. Cosine similarity retrieves the most semantically relevant results.

## Getting Started

### Prerequisites

Python 3.8+

```bash
pip install -r requirements.txt
```

### Run

Pre-computed embeddings and processed data are included in the `data/` folder.

```bash
python main.py
```

## Project Structure

```text
├── data/
│   ├── dataset.pkl
│   └── finalembed.pt
├── onetimescripts/
├── main.py
├── requirements.txt
└── README.md
```

