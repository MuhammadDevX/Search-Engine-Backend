import pandas as pd

df =pd.read_csv("lexicon_dict.csv")

for word in df["Word"]:
  print(word)

print((df.columns))
