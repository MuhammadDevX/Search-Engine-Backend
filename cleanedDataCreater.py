import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer


df = pd.read_csv("test.csv")
df = df.drop_duplicates().dropna()

def process_tags(tags_text):
  try:
    
    tags = tags_text[1:-1].split(", ")
    tags = [w[1:-1] for w in tags]
    return " ".join(tags)
  except Exception as e:
    return ""


if "tags" in df.columns:
  df['tags_text'] = df.tags.apply(process_tags)
else:
  df["tags_text"] = ""
  
  
  
df["merged_text"] = df["title"] + " " + df["text"] + " " + df["tags_text"]


merged_text_df = df[["merged_text"]]
# merged_text_df = merged_text_df.drop_duplicates().dropna()
# merged_text_df.reset_index(drop=True, inplace=True)
# merged_text_df.merged_text = (merged_text_df.merged_text.astype(str))

def clean_characters(text):
  text = re.sub(r'[^\w\s]','',str(text))
  text = text.lower()
  return text

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def clean_stop_words_nltk(text):
    if not isinstance(text, str):  
        return ""
    tokens = word_tokenize(text.lower())  
    clean_list = [lemmatizer.lemmatize(word) for word in tokens if word.isalpha() and word not in stop_words]
    return " ".join(clean_list)

df["merged_text"] = df["merged_text"].apply(clean_characters).apply(clean_stop_words_nltk)



df = df[["merged_text"]].drop_duplicates().reset_index(drop=True)
df.to_csv("cleaned_articles_test.csv",index=False)



