import math
import pandas as pd
import os
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import ast

# Initialize lemmatizer and stopwords
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))

# Load the lexicon and inverted index
lexicon_df = pd.read_csv('lexicon_dict.csv', encoding='utf-8')
lexicon = {row["Word"]: str(row["WordID"]) for _, row in lexicon_df.iterrows()}

# inverted_index_df = pd.read_csv('inverted_indexing_array.csv', encoding='utf-8')
# inverted_index = {
#     str(row["WordID"]): ast.literal_eval(row["Postings"])
#     for _, row in inverted_index_df.iterrows()
# }

# Load document lengths for normalization
forward_index_df = pd.read_csv('forward_indexing.csv', encoding='utf-8')
doc_lengths = {str(row['docID']): len(ast.literal_eval(row['WordOccurrences'])) for _, row in forward_index_df.iterrows()}
avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths)

# BM25 parameters
k1 = 1.5
b = 0.75

# Preprocess the query (lemmatization, stopword removal)
def preprocess_query(query):
    query = query.translate(str.maketrans('', '', string.punctuation)).lower()
    lemmatized_query = [
        lemmatizer.lemmatize(word) for word in query.split() if word not in stop_words
    ]
    return lemmatized_query


def load_word_postings(word_id):
    
    all_postings =[]
    for barrel_file in sorted(os.listdir('barrels/')):
        if barrel_file.startswith('barrel_') and barrel_file.endswith('.csv'):
            barrel_path = os.path.join('barrels/', barrel_file)
            barrel_df = pd.read_csv(barrel_path, encoding='utf-8')
            
            # Find postings for the specific word_id
            word_postings = barrel_df[barrel_df['WordID'] == int(word_id)]
            
            for _, row in word_postings.iterrows():
                # Assuming the postings are stored as a string representation of a list
                postings = ast.literal_eval(row['Postings']) if pd.notna(row['Postings']) else []
                all_postings.extend(postings)
    
    return all_postings
    




# BM25 score calculation
def bm25_score(query_terms, doc_id):
    score = 0
    for term in query_terms:
        if term in lexicon:
            word_id = lexicon[term]
            postings = load_word_postings(word_id)
            
            # Find the posting for the current document
            doc_posting = next((post for post in postings if post["DocID"] == doc_id), None)
            if doc_posting:
                freq = len(doc_posting["Positions"])   
                N = len(doc_lengths)  # Total number of documents
                df = len(postings)   # Document frequency for the term
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)  # IDF component
                
                # BM25 formula
                doc_len = doc_lengths[doc_id]
                tf = freq * (k1 + 1) / (freq + k1 * (1 - b + b * (doc_len / avg_doc_length)))
                score += idf * tf
    return score




# BM25 search
def bm25_search(query, top_n=10):
    query_terms = preprocess_query(query)
    doc_scores = defaultdict(float)

    # Calculate BM25 scores for all documents
    for term in query_terms:
        if term in lexicon:
            word_id = lexicon[term]
            postings = load_word_postings(word_id)
            
            for post in postings:
                doc_id = post["DocID"]
                doc_scores[doc_id] += bm25_score(query_terms, doc_id)

    # Sort by score and return the top N results
    ranked_docs = sorted(doc_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    return ranked_docs

# Example Usage
query = "brain"  # Replace with your test query
results = bm25_search(query)
print(f"Top results for query '{query}':")
for doc_id, score in results:
    print(f"Document {doc_id}: Score {score}")
