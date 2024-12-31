import json
import os
from collections import defaultdict
from typing import Dict, List,Tuple
from dataclasses import dataclass, asdict
from tqdm import tqdm
from preprocessAndLexiconGen import LexiconLoader, DocumentProcessor, LexiconGenerator
from forwardIndexGenerator import ForwardIndexGenerator

@dataclass
class TermPostingData:
    positions: List[int]
    field_counts: Dict[str, int]
    total_freq: int

class InvertedIndexGenerator:
    def __init__(self, lexicon_loader: LexiconLoader):
        self.lexicon = lexicon_loader
        self.target_barrel_size = 1000
    
    def create_posting_entry(self, positions: List[int], field_lengths: Dict[str, int]) -> Dict:
        field_counts = self.count_field_occurrences(positions, field_lengths)
        posting_data = TermPostingData(
            positions=positions,
            field_counts=field_counts,
            total_freq=len(positions)
        )
        return asdict(posting_data)
    
    def count_field_occurrences(self, positions: List[int], field_lengths: Dict[str, int]) -> Dict[str, int]:
        field_counts = defaultdict(int)
        title_end = field_lengths['title']
        text_end = title_end + field_lengths['text']
        
        for pos in positions:
            if pos < title_end:
                field_counts['title'] += 1
            elif pos < text_end:
                field_counts['text'] += 1
            else:
                field_counts['tags'] += 1
                
        return dict(field_counts)
    
    def calculate_barrel_ranges(self, inverted_index: Dict) -> List[Tuple[int, int]]:
        """Calculate barrel ranges based on term frequencies."""
        # Sort terms by document frequency
        term_sizes = [(int(term_id), len(postings)) 
                     for term_id, postings in inverted_index.items()]
        term_sizes.sort(key=lambda x: x[0])
        
        barrel_ranges = []
        current_size = 0
        start_id = term_sizes[0][0]
        
        for term_id, posting_size in term_sizes:
            current_size += posting_size
            if current_size >= self.target_barrel_size:
                barrel_ranges.append((start_id, term_id))
                current_size = 0
                start_id = term_id + 1
        
        # Add remaining terms
        if start_id <= term_sizes[-1][0]:
            barrel_ranges.append((start_id, term_sizes[-1][0]))
        
        return barrel_ranges

    def generate_inverted_index(self, forward_index_path: str, output_dir: str):
        with open(forward_index_path, 'r') as f:
            forward_index = json.load(f)

        inverted_index = defaultdict(dict)
        
        for doc_id, doc_data in tqdm(forward_index.items(), desc="Building inverted index"):
            for word_id, positions in doc_data['terms'].items():
                posting = self.create_posting_entry(positions, doc_data['field_lengths'])
                inverted_index[word_id][doc_id] = posting

        # Calculate dynamic barrel ranges
        barrel_ranges = self.calculate_barrel_ranges(inverted_index)
        os.makedirs(output_dir, exist_ok=True)
        
        # Save barrels using calculated ranges
        for barrel_id, (start_id, end_id) in enumerate(barrel_ranges):
            barrel = {}
            for word_id in range(start_id, end_id + 1):
                word_id_str = str(word_id)
                if word_id_str in inverted_index:
                    barrel[word_id_str] = inverted_index[word_id_str]
            
            barrel_path = os.path.join(output_dir, f"barrel_{barrel_id}.json")
            with open(barrel_path, 'w') as f:
                json.dump(barrel, f)

        metadata = {
            'total_terms': len(inverted_index),
            'barrel_ranges': barrel_ranges,
            'num_barrels': len(barrel_ranges)
        }
        
        with open(os.path.join(output_dir, 'inverted_index_metadata.json'), 'w') as f:
            json.dump(metadata, f)
            
            
def main():
    # Initialize components
    doc_processor = DocumentProcessor()
    processed_docs = doc_processor.process_dataset("test.csv", "processed_articles.json")
    
    # Generate and load lexicon
    lexicon_generator = LexiconGenerator()
    lexicon_dir = "lexicon_output"
    lexicon_generator.generate_lexicon(processed_docs, lexicon_dir)
    lexicon_loader = LexiconLoader(lexicon_dir)
    
    # Generate forward index
    forward_index_generator = ForwardIndexGenerator(lexicon_loader)
    forward_index_path = "forward_index_output/forward_index.json"
    forward_index_generator.generate_forward_index(processed_docs, forward_index_path)
    
    # Generate inverted index
    inverted_generator = InvertedIndexGenerator(lexicon_loader)
    inverted_generator.generate_inverted_index(forward_index_path, "inverted_index_output")

if __name__ == "__main__":
    main()