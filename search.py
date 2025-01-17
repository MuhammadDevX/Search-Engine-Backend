import heapq
from turtle import pos
import pandas as pd
from collections import defaultdict
import json
import os
import numpy as np
from typing import Dict, List
from preprocessAndLexiconGen import LexiconLoader, TextPreprocessor
import time
class QueryProcessor:
    def __init__(self, lexicon, text_preprocessor):
        self.lexicon = lexicon
        self.preprocessor = text_preprocessor
    
    def process_query(self, query: str) -> List[str]:
        # Preprocess the query text
        # start = time.time()
        processed_terms = self.preprocessor.clean_and_lemmatize(self.preprocessor.clean_characters(query)).split()
        # end = time.time()
        # print(f"Time to process the terms is {end - start}")
        # Filter out terms not in lexicon
        valid_terms = [term for term in processed_terms 
                      if self.lexicon.get_word_id(term) != -1]
        return valid_terms

class DocumentRetriever:
    def __init__(self, df):
        self.df = df
        # Ensure all expected columns exist
        required_cols = ['title', 'text', 'url', 'authors', 'timestamp', 'tags']
        for col in required_cols:
            if col not in self.df.columns:
                self.df[col] = ''
    
    def get_document_preview(self, doc_id: str) -> Dict[str, str]:
        try:
            doc = self.df.iloc[int(doc_id)]
            return {
                'title': str(doc['title']),
                'text': str(doc['text'])[:200],
                'url': str(doc['url']),
                'authors': str(doc['authors']),
                'timestamp': str(doc['timestamp']),
                'tags': str(doc['tags'])
            }
        except:
            return {key: '' for key in ['title', 'text', 'url', 'authors', 'timestamp', 'tags']}
    
    def get_full_document(self, doc_id: str) -> Dict[str, str]:
        try:
            doc = self.df.iloc[int(doc_id)]
            return {
                'title': str(doc['title']),
                'text': str(doc['text']),
                'url': str(doc['url']),
                'authors': str(doc['authors']),
                'timestamp': str(doc['timestamp']),
                'tags': str(doc['tags'])
            }
        except:
            return None

class InvertedIndexSearcher:
    def __init__(self, index_dir: str, lexicon):
        self.index_dir = index_dir
        self.lexicon = lexicon
        
        with open(os.path.join(index_dir, 'inverted_index_metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        # print ("I ran as soon as the website loaded or the backend ran")
        self.barrel_ranges = self.metadata['barrel_ranges']
        self.barrel_cache = {}
    
    def _get_barrel_id(self, word_id: int) -> int:
        for i, (start_id, end_id) in enumerate(self.barrel_ranges):
            if start_id <= word_id <= end_id:
                return i
        return -1
    
    def _load_barrel(self, barrel_id: int) -> Dict:
        if barrel_id not in self.barrel_cache:
            barrel_path = os.path.join(self.index_dir, f'barrel_{barrel_id}.json')
            with open(barrel_path, 'r') as f:
                self.barrel_cache = json.load(f)
                print(f"Loading barrel with barrel id {barrel_id}")
        return self.barrel_cache
    
    def get_term_data(self, word: str) -> Dict:
        word_id = self.lexicon.get_word_id(word)
        if word_id == -1:
            return {}
            
        barrel_id = self._get_barrel_id(word_id)
        if barrel_id == -1:
            return {}
            
        barrel = self._load_barrel(barrel_id)
        return barrel.get(str(word_id), {})
    
    def get_positions(self, term_data, doc_id: str) -> List[int]:
        # term_data = self.get_term_data(word)
        return term_data.get(doc_id, {}).get('positions', [])
    
    def get_field_frequencies(self, term_data, doc_id: str) -> Dict[str, int]:
        # term_data = self.get_term_data(word)
        return term_data.get(doc_id, {}).get('field_counts', {})
    
    def get_document_frequency(self, term_data) -> int:
        # term_data = self.get_term_data(word)
        return len(term_data) if term_data else 0
    
    def get_idf(self, term_data) -> float:
        doc_freq = self.get_document_frequency(term_data)
        if doc_freq == 0:
            return 0.0
        return np.log(self.metadata['total_terms'] / (1 + doc_freq))

class SearchEngine:
    def __init__(self, inverted_searcher: InvertedIndexSearcher, 
                 doc_retriever: DocumentRetriever, 
                 query_processor: QueryProcessor):
        self.inverted_searcher = inverted_searcher
        self.doc_retriever = doc_retriever
        self.query_processor = query_processor
        self.field_weights = {
            'title': 3.0,
            'text': 1.0,
            'authors': 1.5,
            'tags': 2.0
        }
    
    def _calculate_bm25_score(self, barrel, doc_id: str) -> float:
        # barrel = self.inverted_searcher.get_term_data(term)
        field_freqs = self.inverted_searcher.get_field_frequencies(barrel, doc_id)
        weighted_tf = sum(freq * self.field_weights.get(field, 1.0)
                         for field, freq in field_freqs.items())
        
        idf = self.inverted_searcher.get_idf(barrel)
        k1, b = 1.5, 0.75
        
        numerator = weighted_tf * (k1 + 1)
        denominator = weighted_tf + k1
        
        return idf * numerator / denominator
    
    def _calculate_proximity_score(self, terms: List[str], doc_id: str, window_size: int = 10) -> float:
        positions = []
        start = time.time()
        for term in terms:
            barrel = self.inverted_searcher.get_term_data(term)
            positions.append(self.inverted_searcher.get_positions(barrel, doc_id))
        end = time.time()
        
        print(f"The time taken for proximity is {end-start}")
        if not all(positions):
            return 0.0
        min_distance = float('inf')
        for positions_combination in zip(*positions):
            distance = max(positions_combination) - min(positions_combination)
            min_distance = min(min_distance, distance)
        
        return 1.0 - min(1.0, min_distance / window_size)
    
    def search(self, query: str, max_results: int = 100) -> List[Dict]:
        # start = time.time()
        processed_terms = self.query_processor.process_query(query)
        # end = time.time()
        # print(f"The time taken to process the terms is {end-start:.4f}")
        doc_scores = defaultdict(float)
        start = time.time()
        for term in processed_terms:
            term_data = self.inverted_searcher.get_term_data(term)
            for doc_id in term_data:
                if doc_id != 'metadata':
                    # Calculate BM25 score
                    doc_scores[doc_id] += self._calculate_bm25_score(term_data, doc_id)
                    
                    # Add temporal boost for newer documents
                    # doc_preview = self.doc_retriever.get_document_preview(doc_id)
        end = time.time()
        
        print(f"the time for  bm25 search is {end-start}")
        # Add proximity boost for multi-term queries
        # start = time.time()
        # if len(processed_terms) > 1:
        #     for doc_id in doc_scores:
        #         proximity_score = self._calculate_proximity_score(processed_terms, doc_id)
        #         doc_scores[doc_id] *= (1 + proximity_score)
        # end = time.time()
        # print(f"The time taken for proximity search is {end-start}")
        top_docs = heapq.nlargest(max_results, doc_scores.items(), key=lambda x: x[1])
        
        results = []
        # start = time.time()
        for doc_id, score in top_docs:
            doc_preview = self.doc_retriever.get_document_preview(doc_id)
            results.append({
                'doc_id': doc_id,
                'title': doc_preview['title'],
                'text': doc_preview['text'],
                'url': doc_preview['url'],
                'authors': doc_preview['authors'],
                'timestamp': doc_preview['timestamp'],
                'tags': doc_preview['tags'],
                'score': score,
                'matched_terms': processed_terms
            })
        # end = time.time()
        
        # print(f"Time taken for rearranging the data is {end-start}")
        return results

# def main():
#     lexicon_loader = LexiconLoader("lexicon_output")
#     text_preprocessor = TextPreprocessor()
#     inverted_searcher = InvertedIndexSearcher("inverted_index_output", lexicon_loader)
#     doc_retriever = DocumentRetriever("test.csv")
    
#     query_processor = QueryProcessor(lexicon_loader, text_preprocessor)
#     search_engine = SearchEngine(inverted_searcher, doc_retriever, query_processor)
    
#     query = "machine learning tutorials and here you go"
#     results = search_engine.search(query)
    
#     print(f"\nResults for query: '{query}'\n" + "="*50)
#     for i, result in enumerate(results, 1):
#         print(f"\n{i}. {result['title']}")
#         print(f"URL: {result['url']}")
#         print(f"Authors: {result['authors']}")
#         print(f"Date: {result['timestamp']}")
#         print(f"Tags: {result['tags']}")
#         print(f"Score: {result['score']:.4f}")
#         print(f"Preview: {result['text']}...")

# if __name__ == "__main__":
#     main()