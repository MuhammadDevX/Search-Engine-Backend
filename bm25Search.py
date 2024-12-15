import math
from turtle import back
import pandas as pd
import os
from collections import defaultdict
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import string
import ast
from globalVariables import BARREL_SIZE
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


forward_index_df = pd.read_csv('forward_indexing.csv', encoding='utf-8')
doc_lengths = {str(row['docID']): len(ast.literal_eval(row['WordOccurrences'])) for _, row in forward_index_df.iterrows()}
avg_doc_length = sum(doc_lengths.values()) / len(doc_lengths)


k1 = 1.5
b = 0.75


def preprocess_query(query):
    query = query.translate(str.maketrans('', '', string.punctuation)).lower()
    lemmatized_query = [
        lemmatizer.lemmatize(word) for word in query.split() if word not in stop_words
    ]
    return lemmatized_query


def get_barrel_number (word_id,barrel_size = BARREL_SIZE):
    return word_id//barrel_size +1

def load_word_postings(word_id):
    all_postings =[]
    barrel_num = get_barrel_number(int(word_id))
    
    barrel_file = f'barrels/barrel_{barrel_num}.csv'
    barrel_df = pd.read_csv(barrel_file, encoding='utf-8')
    word_postings = barrel_df[barrel_df['WordID'] == int(word_id)]
    
    for _, row in word_postings.iterrows():
        postings = ast.literal_eval(row['Postings']) if pd.notna(row['Postings']) else []
        all_postings.extend(postings)
    return all_postings

# this is for score calculation of bm25
def bm25_score(query_terms, doc_id):
    score = 0
    for term in query_terms:
        if term in lexicon:
            word_id = lexicon[term]
            postings = load_word_postings(word_id)
            
           
            doc_posting = next((post for post in postings if post["DocID"] == doc_id), None)
            if doc_posting:
                freq = len(doc_posting["Positions"])   
                N = len(doc_lengths)  
                df = len(postings)   
                idf = math.log((N - df + 0.5) / (df + 0.5) + 1)  
                
                # formula
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
