import pandas as pd
import re
import spacy

nlp = spacy.load("en_core_web_sm")



# so i read the csv
df = pd.read_csv("test.csv")  
df = df.drop_duplicates()
df = df.dropna()



# title text url authors timestamp

non_required_columns =["url","authors","timestamp"]
filtered_df = df.drop(columns=non_required_columns)





def process_tags(tags_text):
  tags = tags_text[1:-1].split(", ")
  tags = [w[1:-1] for w in tags]
  return " ".join(tags)

filtered_df['tags_text'] = filtered_df.tags.apply(process_tags)

filtered_df["merged_text"] = filtered_df["title"] + " " + filtered_df["text"] + " " + filtered_df["tags_text"]


# print(filtered_df["merged_text"])

merged_text_df = filtered_df[["merged_text"]]



merged_text_df.drop_duplicates()
merged_text_df.reset_index(drop=True, inplace=True)
merged_text_df.merged_text = (merged_text_df.merged_text.astype(str))




def clean_characters(text):
  text = re.sub(r'[^\w\s]','',text)
  text = text.lower()
  return text

def clean_stop_words(text):
  doc = nlp(text)
  clean_list = [token.lemma_ for token in doc if not token.is_stop]
  return " ".join(clean_list)

clean_df = pd.DataFrame({"text":[]})
clean_df.text = merged_text_df.merged_text.progress_apply(clean_characters)
clean_df.text = clean_df.text.progress_apply(clean_stop_words)

print(clean_df.head())
  
  
# def lexicon(column):
#   lexicon = set()
#   for doc in column:
#     if doc is not None and type(doc) == str:
#       lexicon.update(doc.split())
#   return lexicon

# lexicon_dict = lexicon(df["title"])
# lexicon_dict.update(lexicon(df["text"]))

# print(lexicon_dict)


# # using a function named tokenize_column whihc would tokenize some specific columns like in muy case there is 
# def tokenize_column_batchwise(column):
#   docs = nlp.pipe(column,batch_size=1000)
#   return [[token.text for token in doc if token.is_alpha] for doc in docs]


# df["title_tokens"] = tokenize_column_batchwise(df["title"])

# print("Tokenizing Titles Completed")
# # df["title_tokens"] = tokenize_column(df["title"])

# # print("Tokenizing Titles Completed")

# # df["text_tokens"] = tokenize_column(df["text"])

# # print(df["text_tokens"])

# # df['timestamp'] = pd.to_datetime(df['timestamp'])

# # print("Converting Timestamps Completed")

# # df.to_csv("processed_medium_articles.csv", index=False)


# # print("Tokenization Completed")

#lexicon become 1 better 2 




# forward 


# docID () : [{1:}]

# docID : [1,2,2,2,2,2,2]

# reverse wordID -> lexicon 
# wordID: [{url:word_weightage}] 
# barrel -> invertedIndex ka chota hissa 
# a-m  m-p 


# wordID -> 