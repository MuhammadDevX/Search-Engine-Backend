import numpy as np
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import json
import math
import os
from preprocessAndLexiconGen import LexiconLoader, DocumentProcessor, LexiconGenerator

class ForwardIndexGenerator:
    def __init__(self, lexicon_loader: LexiconLoader):
        self.lexicon = lexicon_loader
        self.total_docs = 0
        self.average_doc_length = 0
        
    def calculate_bm25_parameters(self, documents: List[Dict]) -> Dict:
        """Calculate corpus-wide statistics for BM25 ranking."""
        doc_lengths = []
        for doc in documents:
            text_length = len(doc['text'].split())
            title_length = len(doc['title'].split()) * 3
            tags_length = len(doc['tags'].split()) * 2
            doc_lengths.append(text_length + title_length + tags_length)
        
        return {
            'total_docs': len(documents),
            'average_doc_length': float(np.mean(doc_lengths)),
            'doc_lengths': doc_lengths
        }
    
    def calculate_term_frequency(self, text: str, field_weight: float = 1.0) -> Dict[int, float]:
        """Calculate weighted term frequencies for a field."""
        term_freq = defaultdict(float)
        words = text.split()
        
        for word in words:
            word_id = self.lexicon.get_word_id(word)
            if word_id != -1:
                term_freq[word_id] += field_weight
                
        return term_freq
    
    def calculate_field_scores(self, doc: Dict) -> Tuple[Dict[int, float], Dict[str, int]]:
        """Calculate weighted term frequencies and field-specific metrics."""
        field_weights = {
            'title': 3.0,
            'text': 1.0,
            'tags': 2.0
        }
        
        combined_freq = defaultdict(float)
        field_lengths = {
            'title': len(doc['title'].split()),
            'text': len(doc['text'].split()),
            'tags': len(doc['tags'].split())
        }
        
        for field, weight in field_weights.items():
            field_freq = self.calculate_term_frequency(doc[field], weight)
            for term_id, freq in field_freq.items():
                combined_freq[term_id] += freq
                
        return combined_freq, field_lengths
    
    def generate_forward_index(self, documents: List[Dict], output_dir: str):
        """Generate forward index and save in multiple JSON files for efficient access."""
        os.makedirs(output_dir, exist_ok=True)
        
        bm25_params = self.calculate_bm25_parameters(documents)
        
        metadata = {
            'corpus_stats': bm25_params,
            'total_documents': len(documents),
            'index_files': []
        }
        
        batch_size = 1000
        current_batch = {}
        
        for i, doc in enumerate(documents, 1):
            doc_id = str(doc['doc_id'])
            
            term_frequencies, field_lengths = self.calculate_field_scores(doc)
            vector_magnitude = math.sqrt(sum(freq * freq for freq in term_frequencies.values()))
            
            doc_entry = {
                'terms': {
                    str(term_id): float(freq)
                    for term_id, freq in term_frequencies.items()
                },
                'field_lengths': field_lengths,
                'vector_magnitude': float(vector_magnitude),
                'total_length': sum(field_lengths.values()),
                'metadata': {
                    'title': doc['title'],
                    'tags': doc['tags'],
                    'url': doc.get('url', ''),  # Added URL to metadata
                    'text': doc['text'][:200]  # Store preview of text
                }
            }
            
            current_batch[doc_id] = doc_entry
            
            if len(current_batch) >= batch_size or i == len(documents):
                batch_num = (i - 1) // batch_size
                batch_file = f"forward_index_batch_{batch_num}.json"
                batch_path = os.path.join(output_dir, batch_file)
                
                with open(batch_path, 'w', encoding='utf-8') as f:
                    json.dump(current_batch, f, indent=2)
                
                metadata['index_files'].append(batch_file)
                current_batch = {}
        
        metadata_path = os.path.join(output_dir, 'forward_index_metadata.json')
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata

class ForwardIndexLoader:
    def __init__(self, index_dir: str, lexicon_loader: LexiconLoader):
        """Load forward index metadata and initialize batch loading."""
        self.index_dir = index_dir
        self.lexicon = lexicon_loader
        
        metadata_path = os.path.join(index_dir, 'forward_index_metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
        
        self.batch_cache = {}
        
    def _load_batch(self, doc_id: str) -> Dict:
        """Load the appropriate batch file for a given document ID."""
        batch_size = 1000
        batch_num = int(doc_id) // batch_size
        batch_file = f"forward_index_batch_{batch_num}.json"
        
        if batch_file not in self.batch_cache:
            batch_path = os.path.join(self.index_dir, batch_file)
            with open(batch_path, 'r', encoding='utf-8') as f:
                self.batch_cache = json.load(f)
        
        return self.batch_cache.get(doc_id, {})
    
    def get_document_terms(self, doc_id: str) -> Dict[int, float]:
        """Get term frequencies for a document."""
        doc_data = self._load_batch(doc_id)
        if doc_data:
            return {int(term_id): freq for term_id, freq in doc_data.get('terms', {}).items()}
        return {}
    
    def get_document_metadata(self, doc_id: str) -> Dict:
        """Get document metadata including URL."""
        doc_data = self._load_batch(doc_id)
        return doc_data.get('metadata', {})
    
    def get_document_url(self, doc_id: str) -> str:
        """Get document URL specifically."""
        doc_data = self._load_batch(doc_id)
        return doc_data.get('metadata', {}).get('url', '')
    
    def get_document_magnitude(self, doc_id: str) -> float:
        """Get document vector magnitude for similarity calculations."""
        doc_data = self._load_batch(doc_id)
        return doc_data.get('vector_magnitude', 0.0)
    
    def get_corpus_stats(self) -> Dict:
        """Get corpus-wide statistics."""
        return self.metadata.get('corpus_stats', {})

    def get_document_size(self, doc_id: str) -> int:
        """Get the total length of a document."""
        doc_data = self._load_batch(doc_id)
        return doc_data.get('total_length', 0)

    def get_average_document_size(self) -> float:
        """Get the average document length in the corpus."""
        return self.metadata.get('corpus_stats', {}).get('average_doc_length', 0.0)

# Example usage
def main():
    """Example of how to use the forward index system."""
    # Initialize components
    doc_processor = DocumentProcessor()
    processed_docs = doc_processor.process_dataset("test.csv", "processed_articles.json")
    
    # Generate lexicon
    lexicon_generator = LexiconGenerator()
    lexicon_dir = "lexicon_output"
    lexicon_data, stats_data = lexicon_generator.generate_lexicon(processed_docs, lexicon_dir)
    
    # Load lexicon
    lexicon_loader = LexiconLoader(lexicon_dir)
    
    # Generate forward index
    forward_index_generator = ForwardIndexGenerator(lexicon_loader)
    index_dir = "forward_index_output"
    metadata = forward_index_generator.generate_forward_index(processed_docs, index_dir)
    
    # Load and use forward index
    index_loader = ForwardIndexLoader(index_dir, lexicon_loader)
    
    # Example: Get information for a document
    doc_id = "1"
    terms = index_loader.get_document_terms(doc_id)
    metadata = index_loader.get_document_metadata(doc_id)
    url = index_loader.get_document_url(doc_id)
    magnitude = index_loader.get_document_magnitude(doc_id)
    corpus_stats = index_loader.get_corpus_stats()
    
    print(f"Document {doc_id} statistics:")
    print(f"Number of terms: {len(terms)}")
    print(f"Vector magnitude: {magnitude}")
    print(f"URL: {url}")
    print(f"Metadata: {metadata}")
    print(f"Corpus statistics: {corpus_stats}")

if __name__ == "__main__":
    main()