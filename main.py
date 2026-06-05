import torch.nn.functional as f 
from transformers import AutoTokenizer, AutoModel 
import torch
import pandas as pd
import time
import sys
df = pd.read_pickle("data/dataset.pkl")
tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
master_vectors = torch.load("data/finalembed.pt", weights_only=True)
matrix_norm = f.normalize(master_vectors, p=2, dim=1)
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

BLUE  = "\033[34m"
PINK  = "\033[35m"
BOLD  = "\033[1m"
RESET = "\033[0m"

logo_lines = [
    r" /$$       /$$ /$$       /$$                                                /$$ /$$          ",
    r"| $$      /$ $$|__/      |__/                                               | $$|__/          ",
    r"| $$ /$$$| $$ /$$| $$   /$$ /$$  /$$$$$$$  /$$$$$$   /$$$$$$   /$$$$$$   /$$$$$$$ /$$  /$$$$$$ ",
    r"| $$/$$ $$ $$| $$| $$  /$$/| $$ /$$_____/ /$$__  $$ /$$__  $$ /$$__  $$ /$$__  $$| $$ |____  $$",
    r"| $$$$_  $$$$| $$| $$$$$$/ | $$|  $$$$$$ | $$  \ $$| $$$$$$$$| $$$$$$$$| $$  | $$| $$  /$$$$$$$",
    r"| $$$/ \  $$$| $$| $$_  $$ | $$ \____  $$| $$  | $$| $$_____/| $$_____/| $$  | $$| $$ /$$__  $$",
    r"| $$/   \  $$| $$| $$ \  $$| $$ /$$$$$$$/| $$$$$$$/|  $$$$$$$|  $$$$$$$|  $$$$$$$| $$|  $$$$$$$",
    r"|__/     \__/|__/|__/  \__/|__/|_______/ | $$____/  \_______/ \_______/ \_______/|__/ \_______/",
    r"                                         | $$                                                 ",
    r"                                         | $$                                                 ",
    r"                                         |__/                                                 "
]

def search_animation():
    BLUE = "\033[34m"
    BOLD = "\033[1m"
    RESET = "\033[0m"
    
    print()
    for _ in range(2): # Repeat the pulse twice
        for dots in [".  ", ".. ", "...", "   "]:
            sys.stdout.write(f"\r{BLUE}{BOLD}SEARCHING{dots}{RESET}")
            sys.stdout.flush()
            time.sleep(0.15)
    print("\n")

def processquery(userquery):
    query_inputs = tokenizer(userquery, padding=True, truncation=True, return_tensors='pt')
    query_mask = query_inputs['attention_mask']
    with torch.no_grad():
        query_output = model(**query_inputs)
        query_embed = query_output[0]
        query_mask_expanded = query_mask.unsqueeze(-1).expand(query_embed.size()).float()
        sum_query = torch.sum(query_embed * query_mask_expanded, dim=1)
        sum_query_mask = query_mask_expanded.sum(1)
        final_query_vector = sum_query / sum_query_mask
        query_norm = f.normalize(final_query_vector, p=2, dim=1)
    return query_norm

def gettop3(matrix_norm, query_norm):
    CYAN   = "\033[36m"
    PINK   = "\033[35m"
    GREEN  = "\033[32m"
    BOLD   = "\033[1m"
    RESET  = "\033[0m"
    pd.set_option('display.max_colwidth', None)
    scores = torch.matmul(matrix_norm, query_norm.T).squeeze()
    topk_scores, topk_indices = torch.topk(scores, k=3)
    print(f"\n{CYAN}=== SEARCH RESULTS ==={RESET}")
    for rank, idx in enumerate(topk_indices):
        row_index = idx.item()
        score = topk_scores[rank].item()
        matching_text = df.iloc[row_index]['text']
        article_title = df.iloc[row_index]['title']
        url = df.iloc[row_index]['url']

        print(f"\n{PINK}[#{rank+1}]{RESET} {BOLD}{article_title}{RESET} (Score: {GREEN}{score:.4f}{RESET})")
        print(f"{matching_text}")
        print(f"{url}")
        print(f"{CYAN}--------------------------------------------------{RESET}")


for i, line in enumerate(logo_lines):
    if i < 4:
        print(BLUE + BOLD + line + RESET)
    else:
        print(PINK + BOLD + line + RESET)
while True:

    user_input = input(f"{PINK}{BOLD}wikispeedia{RESET} {BLUE}❯{RESET} ")
    if user_input.strip().lower() == 'exit':
        print(f"\n\033[31m\033[1mGoodbye!\033[0m\n")
        break
    
    if not user_input.strip():
        print(f"\n\033[31m\033[1mPlease enter a valid search phrase!\033[0m\n")
        continue
    
    search_animation()
    query_vector_normalized = processquery(user_input)
    gettop3(matrix_norm, query_vector_normalized)
