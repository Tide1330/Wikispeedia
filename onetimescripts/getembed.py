from transformers import AutoTokenizer, AutoModel 
import torch
import pandas as pd

tokenizer = AutoTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
print("grabbed tokenizer")
model = AutoModel.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
print("grabbed model")
df = pd.read_pickle("dataset.pkl")
print("read dataset")

allembeds = []
def getembed(index):
    encoded_inputs = tokenizer(df['text'].tolist()[index:index+32], padding=True, truncation=True, return_tensors='pt')
    print("finished tokenizing")
    attention_mask = encoded_inputs['attention_mask']
    print("grabbed attention mask")
    model.eval()
    with torch.no_grad():
        model_output = model(**encoded_inputs)
        print("finished encoding inputs")
        tokenembed = model_output[0]
        mask_expanded = attention_mask.unsqueeze(-1).expand(tokenembed.size()).float()
        print("expanded mask")
        sum_embeddings = torch.sum(tokenembed * mask_expanded, dim = 1)
        print("summed up")
        sum_mask = mask_expanded.sum(1)
        final_embeddings = sum_embeddings / sum_mask
        print("finished embed")
        allembeds.append(final_embeddings)
    currindex = index+32
    return currindex
currentpos = 0 
dflen = len(df)
while currentpos < dflen:
    currentpos = getembed(currentpos)

final_vector = torch.cat(allembeds, dim=0)
print(f"shape: {final_vector.shape}")
torch.save(final_vector, "finalembed.pt")
print("Saved tensor successfully!")

