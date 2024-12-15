import pandas as pd

# so i read the csv
df = pd.read_csv("cleaned_articles_test.csv")  
df = df.drop_duplicates().dropna()

# title text url authors timestamp
non_required_columns =["url","authors","timestamp"]
filtered_df = df.drop(columns=[col for col in non_required_columns if col in df.columns])


# print(filtered_df.columns)
def create_lexicon_and_hits(data):
  lexicon = {}
  hits = []
  word_id = 0
  for doc_id,doc in enumerate(data,start=1):
    if doc is not None and isinstance(doc,str):
      words = doc.split()
      for position,word in enumerate(words):
        if word not in lexicon:
          lexicon[word] = word_id
          word_id += 1
        hit_type = "plain"
        if position <= 3:
          hit_type = "fancy"
        
        hits.append((lexicon[word],doc_id,position,hit_type))
  return lexicon,hits

lexicon_dict,hits_list = create_lexicon_and_hits(filtered_df["merged_text"])



# saving as a csv
df = pd.DataFrame(list(lexicon_dict.items()),columns=["Word","WordID"])
df = df.sort_values(by="WordID")
df.to_csv("lexicon_dict.csv",index=False,encoding = "utf-8")


hits_df = pd.DataFrame(hits_list,columns=["WordID","DocID","Position","HitType"]).sort_values(by=["WordID","DocID","Position"])
hits_df.to_csv("hits.csv",index=False,encoding = "utf-8")
print("Lexicon and hits saved to 'lexicon_dict.csv' and 'hits.csv'")