import heapq
import pandas as pd
from collections import defaultdict
import json
import os
import numpy as np
from typing import Dict, List
from preprocessAndLexiconGen import LexiconLoader, TextPreprocessor

class QueryProcessor:
    def __init__(self, lexicon_loader, text_preprocessor):
        self.lexicon = lexicon_loader
        self.preprocessor = text_preprocessor
    
    def process_query(self, query: str) -> List[str]:
        # Preprocess the query text
        processed_terms = self.preprocessor.clean_and_lemmatize(self.preprocessor.clean_characters(query)).split()
        # Filter out terms not in lexicon
        valid_terms = [term for term in processed_terms 
                      if self.lexicon.get_word_id(term) != -1]
        return valid_terms

class DocumentRetriever:
    def __init__(self, dataset_path: str):
        self.df = pd.read_csv(dataset_path)
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
    def __init__(self, index_dir: str, lexicon_loader: LexiconLoader):
        self.index_dir = index_dir
        self.lexicon = lexicon_loader
        
        with open(os.path.join(index_dir, 'inverted_index_metadata.json'), 'r') as f:
            self.metadata = json.load(f)
        
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
    
    def get_positions(self, word: str, doc_id: str) -> List[int]:
        term_data = self.get_term_data(word)
        return term_data.get(doc_id, {}).get('positions', [])
    
    def get_field_frequencies(self, word: str, doc_id: str) -> Dict[str, int]:
        term_data = self.get_term_data(word)
        return term_data.get(doc_id, {}).get('field_counts', {})
    
    def get_document_frequency(self, word: str) -> int:
        term_data = self.get_term_data(word)
        return len(term_data) if term_data else 0
    
    def get_idf(self, word: str) -> float:
        doc_freq = self.get_document_frequency(word)
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
    
    def _calculate_bm25_score(self, term: str, doc_id: str) -> float:
        field_freqs = self.inverted_searcher.get_field_frequencies(term, doc_id)
        weighted_tf = sum(freq * self.field_weights.get(field, 1.0)
                         for field, freq in field_freqs.items())
        
        idf = self.inverted_searcher.get_idf(term)
        k1, b = 1.5, 0.75
        
        numerator = weighted_tf * (k1 + 1)
        denominator = weighted_tf + k1
        
        return idf * numerator / denominator
    
    def _calculate_proximity_score(self, terms: List[str], doc_id: str, window_size: int = 10) -> float:
        positions = [self.inverted_searcher.get_positions(term, doc_id) for term in terms]
        if not all(positions):
            return 0.0
            
        min_distance = float('inf')
        for positions_combination in zip(*positions):
            distance = max(positions_combination) - min(positions_combination)
            min_distance = min(min_distance, distance)
        
        return 1.0 - min(1.0, min_distance / window_size)
    
    def search(self, query: str, max_results: int = 10) -> List[Dict]:
        processed_terms = self.query_processor.process_query(query)
        doc_scores = defaultdict(float)
        
        for term in processed_terms:
            term_data = self.inverted_searcher.get_term_data(term)
            for doc_id in term_data:
                if doc_id != 'metadata':
                    # Calculate BM25 score
                    doc_scores[doc_id] += self._calculate_bm25_score(term, doc_id)
                    
                    # Add temporal boost for newer documents
                    doc_preview = self.doc_retriever.get_document_preview(doc_id)
        
        # Add proximity boost for multi-term queries
        if len(processed_terms) > 1:
            for doc_id in doc_scores:
                proximity_score = self._calculate_proximity_score(processed_terms, doc_id)
                doc_scores[doc_id] *= (1 + proximity_score)
        
        top_docs = heapq.nlargest(max_results, doc_scores.items(), key=lambda x: x[1])
        
        results = []
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
        
        return results

def main():
    lexicon_loader = LexiconLoader("lexicon_output")
    text_preprocessor = TextPreprocessor()
    inverted_searcher = InvertedIndexSearcher("inverted_index_output", lexicon_loader)
    doc_retriever = DocumentRetriever("test.csv")
    
    query_processor = QueryProcessor(lexicon_loader, text_preprocessor)
    search_engine = SearchEngine(inverted_searcher, doc_retriever, query_processor)
    
    query = "machine learning tutorials and here you go"
    results = search_engine.search(query)
    
    print(f"\nResults for query: '{query}'\n" + "="*50)
    for i, result in enumerate(results, 1):
        print(f"\n{i}. {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Authors: {result['authors']}")
        print(f"Date: {result['timestamp']}")
        print(f"Tags: {result['tags']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Preview: {result['text']}...")

if __name__ == "__main__":
    main()