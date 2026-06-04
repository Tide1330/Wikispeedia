import pickle
import pandas as pd

with open("dataset.pkl", "rb") as file:
    df = pickle.load(file)
print("Opened training dataset")

#delete references section and also split every 150 words so we can fit it inside our model
df['text'] = df['text'].str.split('References').str[0]
df['text'] = df['text'].str.findall(r'.{1,150}')
df = df.explode('text').reset_index(drop=True)

with open("dataset.pkl","wb") as file:
    pickle.dump(df, file)
print("Data cleaned!")



    

