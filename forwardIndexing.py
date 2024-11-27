import csv
from collections import defaultdict
import string

# Initialize a dictionary to store the forward index
forward_index = defaultdict(list)

# Load lexicon (wordID and word)
lexicon = {}
with open('D:\\DSA project\\Search-Engine-Backend\\lexicon_dict.csv', mode='r', encoding='utf-8') as lexicon_file:
    lexicon_reader = csv.reader(lexicon_file)
    next(lexicon_reader)  # Skip header if exists
    for row in lexicon_reader:
        word_id, word = row
        lexicon[word.lower()] = int(word_id)  # Store lowercase word to ensure case-insensitivity

# Function to preprocess text: remove punctuation, convert to lowercase
def preprocess_text(text):
    # Remove punctuation and convert to lowercase
    return text.translate(str.maketrans('', '', string.punctuation)).lower().split()

# Process metadata.csv and build forward index
with open('D:\\DSA project\\Search-Engine-Backend\\medium_articles.csv', mode='r', encoding='utf-8') as metadata_file:
    metadata_reader = csv.DictReader(metadata_file)
    for row in metadata_reader:
        article_id = row['url']  # Assuming 'url' is used as the unique identifier for each article
        text = row['text']  # Text of the article
        
        # Preprocess the text to remove punctuation and split into words
        words_in_article = preprocess_text(text)
        
        # Add word IDs to the forward index
        for word in words_in_article:
            if word in lexicon:
                word_id = lexicon[word]
                forward_index[word_id].append(article_id)

# Save the forward index to a CSV file
with open('D:\\DSA project\\Search-Engine-Backend\\forward_indexing.csv', mode='w', encoding='utf-8', newline='') as forward_index_file:
    writer = csv.writer(forward_index_file)
    writer.writerow(['WordID', 'ArticleURLs'])  # Write header
    
    # Sort WordIDs and process each word's articles
    for word_id in sorted(forward_index.keys()):  # Sorting WordIDs
        articles = sorted(set(forward_index[word_id]))  # Remove duplicates and sort article URLs
        writer.writerow([word_id, ', '.join(articles)])  # Write WordID and sorted article URLs

print("Forward index saved to 'forward_indexing.csv'")
