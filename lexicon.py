import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
# nltk.download("punkt")
# nltk.download("punkt_tab")
# nltk.download("stopwords")
# nltk.download("wordnet")


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


merged_text_df = filtered_df[["merged_text"]]



merged_text_df.drop_duplicates()
merged_text_df.reset_index(drop=True, inplace=True)
merged_text_df.merged_text = (merged_text_df.merged_text.astype(str))




def clean_characters(text):
  text = re.sub(r'[^\w\s]','',text)
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




# filtered_df["merged_text"] = filtered_df.merged_text.apply(clean_characters)
filtered_df["merged_text"] = filtered_df.merged_text.apply(clean_stop_words_nltk)




# clean_df.text = clean_characters(merged_text_df.merged_text)
# clean_df.text = clean_df.text.apply(clean_stop_words)

# print(filtered_df["merged_text"])
  
  
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
