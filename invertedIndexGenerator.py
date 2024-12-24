import json
import os
import math
from typing import Dict, List, Set, Tuple
from collections import defaultdict
import numpy as np
import hashlib
from preprocessAndLexiconGen import LexiconLoader, DocumentProcessor, LexiconGenerator

class Barrel:
    def __init__(self, barrel_id: str):
        self.barrel_id = barrel_id
        self.terms = defaultdict(lambda: {
            'document_frequencies': defaultdict(float),
            'field_frequencies': defaultdict(lambda: defaultdict(int)),
            'positions': defaultdict(list),
            'idf': 0.0,
            'total_frequency': 0
        })
    
    def add_term(self, term_id: int, term_data: Dict):
        """Add term data to the barrel."""
        self.terms[term_id] = term_data
    
    def get_term_data(self, term_id: int) -> Dict:
        """Get term data from the barrel."""
        return self.terms.get(term_id, {})
    
    def to_dict(self) -> Dict:
        """Convert barrel data to serializable dictionary."""
        return {
            str(term_id): {
                'document_frequencies': {
                    str(doc_id): float(freq)
                    for doc_id, freq in term_data['document_frequencies'].items()
                },
                'field_frequencies': {
                    str(doc_id): {
                        field: count
                        for field, count in field_freqs.items()
                    }
                    for doc_id, field_freqs in term_data['field_frequencies'].items()
                },
                'positions': {
                    str(doc_id): positions
                    for doc_id, positions in term_data['positions'].items()
                },
                'idf': float(term_data['idf']),
                'total_frequency': int(term_data['total_frequency'])
            }
            for term_id, term_data in self.terms.items()
        }

class BarrelManager:
    def __init__(self, num_barrels: int = 256):
        """Initialize barrel manager with specified number of barrels."""
        self.num_barrels = num_barrels
        self.barrels = {i: Barrel(str(i)) for i in range(num_barrels)}
    
    def get_barrel_id(self, term_id: int) -> int:
        """Get barrel ID for a term using modulo hashing."""
        return term_id % self.num_barrels
    
    def add_term(self, term_id: int, term_data: Dict):
        """Add term data to appropriate barrel."""
        barrel_id = self.get_barrel_id(term_id)
        self.barrels[barrel_id].add_term(term_id, term_data)
    
    def get_term_data(self, term_id: int) -> Dict:
        """Get term data from appropriate barrel."""
        barrel_id = self.get_barrel_id(term_id)
        return self.barrels[barrel_id].get_term_data(term_id)
    
    def save_barrels(self, output_dir: str):
        """Save all barrels to disk."""
        for barrel_id, barrel in self.barrels.items():
            barrel_file = f"barrel_{barrel_id}.json"
            with open(os.path.join(output_dir, barrel_file), 'w', encoding='utf-8') as f:
                json.dump(barrel.to_dict(), f, indent=2)

class InverseIndexGenerator:
    def __init__(self, lexicon_loader: LexiconLoader, num_barrels: int = 256):
        self.lexicon = lexicon_loader
        self.barrel_manager = BarrelManager(num_barrels)
        self.index = defaultdict(lambda: {
            'document_frequencies': defaultdict(float),
            'field_frequencies': defaultdict(lambda: defaultdict(int)),
            'positions': defaultdict(list),
            'idf': 0.0,
            'total_frequency': 0
        })
        
    def process_document_terms(self, doc_id: str, doc: Dict, field: str, weight: float = 1.0):
        """Process terms from a specific field of a document."""
        words = doc[field].split()
        
        for position, word in enumerate(words):
            word_id = self.lexicon.get_word_id(word)
            if word_id != -1:
                self.index[word_id]['document_frequencies'][doc_id] += weight
                self.index[word_id]['field_frequencies'][doc_id][field] += 1
                self.index[word_id]['positions'][doc_id].append(position)
                self.index[word_id]['total_frequency'] += 1
    
    def calculate_idf_scores(self, total_documents: int):
        """Calculate IDF scores for all terms."""
        for term_id, term_data in self.index.items():
            doc_count = len(term_data['document_frequencies'])
            term_data['idf'] = math.log((total_documents + 1) / (doc_count + 0.5))
    
    def generate_inverse_index(self, documents: List[Dict], output_dir: str):
        """Generate inverse index and distribute across barrels."""
        os.makedirs(output_dir, exist_ok=True)
        
        field_weights = {
            'title': 3.0,
            'text': 1.0,
            'tags': 2.0
        }
        
        # Process documents
        for doc in documents:
            doc_id = str(doc['doc_id'])
            for field, weight in field_weights.items():
                self.process_document_terms(doc_id, doc, field, weight)
        
        # Calculate IDF scores
        self.calculate_idf_scores(len(documents))
        
        # Distribute terms across barrels
        for term_id, term_data in self.index.items():
            self.barrel_manager.add_term(term_id, term_data)
        
        # Save barrels
        self.barrel_manager.save_barrels(output_dir)
        
        # Save metadata
        metadata = {
            'total_terms': len(self.index),
            'total_documents': len(documents),
            'num_barrels': self.barrel_manager.num_barrels
        }
        
        with open(os.path.join(output_dir, 'inverse_index_metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(metadata, f, indent=2)
        
        return metadata

class InverseIndexLoader:
    def __init__(self, index_dir: str, lexicon_loader: LexiconLoader):
        """Initialize inverse index loader with barrel loading capability."""
        self.index_dir = index_dir
        self.lexicon = lexicon_loader
        self.barrel_cache = {}
        
        # Load metadata
        metadata_path = os.path.join(index_dir, 'inverse_index_metadata.json')
        with open(metadata_path, 'r', encoding='utf-8') as f:
            self.metadata = json.load(f)
            self.num_barrels = self.metadata['num_barrels']
    
    def _get_barrel_id(self, term_id: int) -> int:
        """Get barrel ID for a term."""
        return term_id % self.num_barrels
    
    def _load_barrel(self, barrel_id: int) -> Dict:
        """Load a barrel from disk if not in cache."""
        if barrel_id not in self.barrel_cache:
            barrel_file = f"barrel_{barrel_id}.json"
            barrel_path = os.path.join(self.index_dir, barrel_file)
            with open(barrel_path, 'r', encoding='utf-8') as f:
                self.barrel_cache[barrel_id] = json.load(f)
        return self.barrel_cache[barrel_id]
    
    def get_term_data(self, term: str) -> Dict:
        """Get all data for a given term."""
        term_id = self.lexicon.get_word_id(term)
        if term_id == -1:
            return {}
            
        barrel_id = self._get_barrel_id(term_id)
        barrel_data = self._load_barrel(barrel_id)
        return barrel_data.get(str(term_id), {})
    
    def get_document_frequency(self, term: str, doc_id: str) -> float:
        """Get frequency of term in specific document."""
        term_data = self.get_term_data(term)
        return term_data.get('document_frequencies', {}).get(str(doc_id), 0.0)
    
    def get_field_frequencies(self, term: str, doc_id: str) -> Dict[str, int]:
        """Get field-specific frequencies for a term in a document."""
        term_data = self.get_term_data(term)
        return term_data.get('field_frequencies', {}).get(str(doc_id), {})
    
    def get_positions(self, term: str, doc_id: str) -> List[int]:
        """Get positions of term in specific document."""
        term_data = self.get_term_data(term)
        return term_data.get('positions', {}).get(str(doc_id), [])
    
    def get_idf(self, term: str) -> float:
        """Get IDF score for a term."""
        term_data = self.get_term_data(term)
        return term_data.get('idf', 0.0)

def main():
    """Example usage of the barrel-based inverse index system."""
    # Initialize components
    doc_processor = DocumentProcessor()
    processed_docs = doc_processor.process_dataset("test.csv", "processed_articles.json")
    
    # Generate lexicon
    lexicon_generator = LexiconGenerator()
    lexicon_dir = "lexicon_output"
    lexicon_data, stats_data = lexicon_generator.generate_lexicon(processed_docs, lexicon_dir)
    
    # Load lexicon
    lexicon_loader = LexiconLoader(lexicon_dir)
    
    # Generate inverse index with barrels
    inverse_index_generator = InverseIndexGenerator(lexicon_loader, num_barrels=256)
    index_dir = "inverse_index_output"
    metadata = inverse_index_generator.generate_inverse_index(processed_docs, index_dir)
    
    # Load and use inverse index
    index_loader = InverseIndexLoader(index_dir, lexicon_loader)
    
    # Example usage
    term = "example"
    doc_id = "1"
    
    # Get term statistics
    term_data = index_loader.get_term_data(term)
    doc_freq = index_loader.get_document_frequency(term, doc_id)
    field_freqs = index_loader.get_field_frequencies(term, doc_id)
    positions = index_loader.get_positions(term, doc_id)
    idf = index_loader.get_idf(term)
    
    print(f"Statistics for term '{term}' in document {doc_id}:")
    print(f"Document frequency: {doc_freq}")
    print(f"Field frequencies: {field_freqs}")
    print(f"Positions: {positions}")
    print(f"IDF score: {idf}")

if __name__ == "__main__":
    main()