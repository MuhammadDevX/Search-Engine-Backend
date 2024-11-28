import pandas as pd

# so i read the csv
df = pd.read_csv("cleaned_articles_test.csv")  
df = df.drop_duplicates()
df = df.dropna()

# title text url authors timestamp
non_required_columns =["url","authors","timestamp"]
filtered_df = df.drop(columns=non_required_columns)
  
def lexicon(column):
  index = 0
  dictionary ={}
  for doc in column:
    if doc is not None and type(doc) == str:
      for word in doc.split():
        if word not in dictionary.values():
          dictionary[str(index)] = word
          index+=1
  return dictionary

lexicon_dict = lexicon(filtered_df["merged_text"])

# saving as a csv
df = pd.DataFrame(list(lexicon_dict.items()),columns=["WordID","Word"])
df.to_csv("lexicon_dict.csv")
