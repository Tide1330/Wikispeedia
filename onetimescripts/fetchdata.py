#DW about this script I only ran it once to stream all the data I needed to my computer
import pandas as pd
from datasets import load_dataset
import pickle
streamed_dataset = load_dataset("olm/olm-wikipedia-20221220", split="train", streaming=True)
dataset = streamed_dataset.take(1000)
df = pd.DataFrame(list(dataset))
with open("dataset.pkl", "wb") as file:
    pickle.dump(df, file)

