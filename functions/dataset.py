import pandas as pd


df = pd.read_csv("../medium_articles.csv")

df = df[1:20]

# now saving this dataframe

df.to_csv("test.csv")
