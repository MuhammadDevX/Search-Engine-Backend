import numpy as np
from typing import Dict, List
from collections import defaultdict
import json
import os
from preprocessAndLexiconGen import LexiconLoader,DocumentProcessor,LexiconGenerator

class ForwardIndexGenerator:
    def __init__(self, lexicon_loader: LexiconLoader):
        self.lexicon = lexicon_loader
        
    def index_field(self, text: str, field: str, position_offset: int = 0) -> Dict[int, List[int]]:
        """Index a field's terms with position information."""
        positions = defaultdict(list)
        words = text.split()
        
        for pos, word in enumerate(words):
            word_id = self.lexicon.get_word_id(word)
            if word_id != -1:
                positions[word_id].append(position_offset + pos)
        
        return positions
    
    def index_document(self, doc: Dict) -> Dict:
        """Create forward index entry for a document."""
        # Calculate position offsets
        title_len = len(doc['title'].split())
        text_len = len(doc['text'].split())
        
        # Index each field with appropriate offsets
        title_positions = self.index_field(doc['title'], 'title', 0)
        text_positions = self.index_field(doc['text'], 'text', title_len)
        tags_positions = self.index_field(doc['tags'], 'tags', title_len + text_len)
        
        # Merge positions
        all_positions = defaultdict(list)
        for positions in [title_positions, text_positions, tags_positions]:
            for word_id, pos_list in positions.items():
                all_positions[word_id].extend(pos_list)
        
        return {
            'terms': {str(word_id): positions for word_id, positions in all_positions.items()},
            'field_lengths': {
                'title': title_len,
                'text': text_len,
                'tags': len(doc['tags'].split())
            },
            'total_length': title_len + text_len + len(doc['tags'].split()),
            'metadata': {
                'title': doc['title'],
                'text_preview': doc['text'][:200],
                'tags': doc['tags']
            }
        }
    
    def generate_forward_index(self, documents: List[Dict], output_file: str) -> Dict:
        """Generate complete forward index."""
        forward_index = {}
        
        for doc in documents:
            doc_id = str(doc['doc_id'])
            forward_index[doc_id] = self.index_document(doc)
        
        # Save to single file
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(forward_index, f, indent=2)
        
        return forward_index

def main():
    # Initialize and process documents
    doc_processor = DocumentProcessor()
    processed_docs = doc_processor.process_dataset("test.csv", "processed_articles.json")
    
    # Generate lexicon
    lexicon_generator = LexiconGenerator()
    lexicon_dir = "lexicon_output"
    lexicon_generator.generate_lexicon(processed_docs, lexicon_dir)
    
    # Create forward index
    lexicon_loader = LexiconLoader(lexicon_dir)
    index_generator = ForwardIndexGenerator(lexicon_loader)
    forward_index = index_generator.generate_forward_index(
        processed_docs,
        "forward_index_output/forward_index.json"
    )
    
    # Print basic statistics
    print(f"Indexed {len(forward_index)} documents")
    sample_doc = next(iter(forward_index.values()))
    print(f"Sample document terms: {len(sample_doc['terms'])}")
    print(f"Sample field lengths: {sample_doc['field_lengths']}")

if __name__ == "__main__":
    main()