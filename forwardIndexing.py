import pandas as pd
import string
from collections import defaultdict

# loading the lexicon
lexicon_df = pd.read_csv('lexicon_dict.csv', encoding='utf-8')
lexicon = { row["Word"]:str(row["WordID"]) for _, row in lexicon_df.iterrows()}

# function for preprocessing text
def preprocess_text(text):
    return text.translate(str.maketrans('', '', string.punctuation)).lower().split()

# loadint the cleaned articles
articles_df = pd.read_csv('cleaned_articles_test.csv')
forward_index = defaultdict(list)

# build the forward index
for index, row in articles_df.iterrows():
    text = row['merged_text']
    words_in_article = preprocess_text(text)

    # create the hashmap
    for word in words_in_article:
        if word in lexicon:
            word_id = lexicon[word]
            forward_index[str(index)].append(int(word_id))  

# create a dataframe from the forward index hashmap
forward_index_df = pd.DataFrame({
    'docID': forward_index.keys(),
    'WordIDs': [(sorted((word_ids))) for word_ids in forward_index.values()]  
})

# save csv
forward_index_df.to_csv('forward_indexing.csv', index = False, encoding = 'utf-8')
print("Forward index saved to 'forward_indexing.csv'")
