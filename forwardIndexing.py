import pandas as pd
import string
from collections import defaultdict

# Load lexicon into a dictionary
lexicon_df = pd.read_csv('lexicon_dict.csv', encoding='utf-8')
lexicon = { row["Word"]:str(row["WordID"]) for _, row in lexicon_df.iterrows()}

# Preprocessing function to clean text
def preprocess_text(text):
    return text.translate(str.maketrans('', '', string.punctuation)).lower().split()

# Load articles and initialize forward index
articles_df = pd.read_csv('cleaned_articles_test.csv')
forward_index = defaultdict(list)

# Build the forward index
for index, row in articles_df.iterrows():
    # Extract and preprocess text
    text = row['merged_text']
    words_in_article = preprocess_text(text)

    # Map words to word IDs and build the index
    for word in words_in_article:
        if word in lexicon:
            word_id = lexicon[word]
            forward_index[str(index)].append(str(word_id))  # Use the article index as docID

# Prepare forward index for saving
forward_index_df = pd.DataFrame({
    'docID': forward_index.keys(),
    'WordIDs': [(sorted((word_ids))) for word_ids in forward_index.values()]  
})

# Save forward index to CSV
forward_index_df.to_csv('forward_indexing.csv', index = False, encoding = 'utf-8')
print("Forward index saved to 'forward_indexing.csv'")
