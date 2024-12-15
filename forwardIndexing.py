import pandas as pd
import string
from collections import defaultdict

# loading the lexicon
lexicon_df = pd.read_csv('lexicon_dict.csv', encoding='utf-8')
lexicon = { row["Word"]:int(row["WordID"]) for _, row in lexicon_df.iterrows()}

articles_df = pd.read_csv('cleaned_articles_test.csv')

# function for preprocessing text
def preprocess_text(text):
    return text.translate(str.maketrans('', '', string.punctuation)).lower().split()


forward_index = defaultdict(list)

# build the forward index
for doc_id, row in articles_df.iterrows():
    text = row['merged_text']
    words_in_article = preprocess_text(text)

    # create the hashmap
    for position, word in enumerate(words_in_article):
        if word in lexicon:
            word_id = lexicon[word]
            hit_type = "plain"
            if position <= 3:
                hit_type = "fancy"
            
            forward_index[str(doc_id)].append((word_id,position,hit_type)) 

# create a dataframe from the forward index hashmap
forward_index_df = pd.DataFrame({
    'docID': forward_index.keys(),
    "WordOccurrences": [
        sorted(word_occurrences, key=lambda x: (x[0], x[1])) for word_occurrences in forward_index.values()
    ]
})



# save csv
forward_index_df.to_csv('forward_indexing.csv', index = False, encoding = 'utf-8')
print("Forward index with proximity and hits saved to 'forward_indexing.csv'")
