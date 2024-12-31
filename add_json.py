import json
import os
import csv
from pydoc import text
from typing import Dict, Any, Optional
from collections import defaultdict
from dataclasses import dataclass, asdict
from datetime import datetime
from preprocessAndLexiconGen import TextPreprocessor

class PathManager:
    """Manages paths for index components."""
    
    def __init__(self, base_dir: str = "search_index"):
        self.base_dir = base_dir
        self.lexicon_dir = os.path.join(base_dir, "lexicon_output")
        self.forward_index_dir = os.path.join(base_dir, "forward_index_output")
        self.inverse_index_dir = os.path.join(base_dir, "inverted_index_output")
        
        # Create necessary directories
        for directory in [self.base_dir, self.lexicon_dir, 
                         self.forward_index_dir, self.inverse_index_dir]:
            os.makedirs(directory, exist_ok=True)
    
    @property
    def paths(self) -> Dict[str, str]:
        return {
            'lexicon': os.path.join(self.lexicon_dir, 'lexicon.json'),
            'lexicon_stats': os.path.join(self.lexicon_dir, 'lexicon_stats.json'),
            'forward_index_metadata': os.path.join(self.forward_index_dir, 'forward_index_metadata.json'),
            'forward_index': os.path.join(self.forward_index_dir, 'forward_index.json'),
            'inverse_index_metadata': os.path.join(self.inverse_index_dir, 'inverted_index_metadata.json')
        }

class DocumentValidator:
    """Validates document format for indexing."""
    
    def validate_document(self, doc: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        required_fields = {'doc_id', 'title', 'text', 'tags'}
        
        # Check required fields
        if missing := required_fields - set(doc.keys()):
            return False, f"Missing required fields: {missing}"
        
        # Validate doc_id
        if not isinstance(doc['doc_id'], (int, str)):
            return False, "doc_id must be an integer or string"
        
        # Validate string fields
        for field in ['title', 'text', 'tags']:
            if not isinstance(doc[field], str):
                return False, f"{field} must be a string"
            if not doc[field].strip():
                return False, f"{field} cannot be empty"
        
        return True, None




@dataclass
class TermPostingData:
    positions: list[int]
    field_counts: Dict[str, int]
    total_freq: int

class DocumentAdder:
    def __init__(self):
        self.path_manager = PathManager("./")
        self.validator = DocumentValidator()
        self.csv_path = "test.csv"
   
    
    def _load_json(self, path: str, default: Any = None) -> Any:
        if os.path.exists(path):
            with open(path, 'r', encoding='utf-8') as f:
                return json.load(f)
        return default if default is not None else {}
    
    def _save_json(self, data: Any, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2)
    
    def _append_to_csv(self, doc: Dict[str, Any]) -> bool:
        """Append the document to the CSV file with specific fields."""
        try:
            # Read existing CSV to get the last line
            last_line = None
            with open(self.csv_path, 'r', encoding='utf-8') as csvfile:
                for line in csvfile:
                    if line.strip():  # Skip empty lines
                        last_line = line
            
            # Append the new document
            with open(self.csv_path, 'a', newline='', encoding='utf-8') as csvfile:
                fieldnames = ['title', 'text', 'url', 'authors', 'timestamp', 'tags']
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                
                # Add a newline if the last line doesn't end with one
                if last_line and not last_line.endswith('\n'):
                    csvfile.write('\n')
                
                # Prepare the row with all required fields
                row = {
                    'title': doc.get('title', ''),
                    'text': doc.get('text', ''),
                    'url': doc.get('url', f'https://example.com/doc/{doc["doc_id"]}'),  # Default URL
                    'authors': doc.get('authors', ''),  # Empty string if not provided
                    'timestamp': doc.get('timestamp', datetime.now().strftime('%Y-%m-%d %H:%M:%S')),
                    'tags': doc.get('tags', '')
                }
                
                writer.writerow(row)
            return True
        except Exception as e:
            print(f"Error appending to CSV: {str(e)}")
            return False
            
    def _get_barrel_id(self, term_id: int, metadata: Dict) -> int:
        """Find the appropriate barrel for a term based on barrel ranges."""
        barrel_ranges = metadata.get('barrel_ranges', [])
        # print(f"barrel_ranges: {barrel_ranges}")
        num_barrels = metadata.get('num_barrels', 0)
        # print(f"num_barrels: {num_barrels}")
        for i, (start, end) in enumerate(barrel_ranges):
            if start <= term_id <= end:
                return i
        return num_barrels  # Default to new barrel if not found
    
    def _get_barrel_path(self, barrel_id: int) -> str:
        return os.path.join(self.path_manager.inverse_index_dir, f"barrel_{barrel_id}.json")
    
    def _count_field_occurrences(self, positions: list[int], field_lengths: Dict[str, int]) -> Dict[str, int]:
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
    
    def _create_posting_entry(self, positions: list[int], field_lengths: Dict[str, int]) -> Dict:
        field_counts = self._count_field_occurrences(positions, field_lengths)
        posting_data = TermPostingData(
            positions=positions,
            field_counts=field_counts,
            total_freq=len(positions)
        )
        return asdict(posting_data)

    def add_document(self, doc: Dict[str, Any]) -> tuple[bool, Optional[str]]:
        """Add a new document to all index components and CSV file."""
        if not (is_valid := self.validator.validate_document(doc))[0]:
            return is_valid
        
        try:
            # First append to CSV
            if not self._append_to_csv(doc):
                print("Failed to append to CSV")
                return False, "Failed to append document to CSV file"
            print("Appended to CSV")
            doc_id = str(doc['doc_id'])
            
            # Load metadata
            metadata = self._load_json(self.path_manager.paths['inverse_index_metadata'])
            
            # Update lexicon
            lexicon = self._load_json(self.path_manager.paths['lexicon'])
            term_mapping = {}
            
            # Process document terms
            document_terms = defaultdict(list)  # term_id -> positions
            field_lengths = {field: len(doc[field].split()) for field in ['title', 'text', 'tags']}
            
            position = 0
            text_preprocessor = TextPreprocessor()
            for field in ['title', 'text', 'tags']:
                words = text_preprocessor.clean_and_lemmatize(text_preprocessor.clean_characters(doc[field])).split()
                for word in words:
                    if word not in lexicon:
                        print(f"Adding new term to lexicon: {word}")
                        lexicon[word] = len(lexicon)
                    term_id = lexicon[word]
                    term_mapping[word] = term_id
                    document_terms[term_id].append(position)
                    position += 1
            
            # Save updated lexicon
            self._save_json(lexicon, self.path_manager.paths['lexicon'])
            
            
            forward_batch = self._load_json(self.path_manager.paths['forward_index'])
            forward_batch[doc_id] = {
                'terms': {str(term_id): positions for term_id, positions in document_terms.items()},
                'field_lengths': field_lengths,
                'metadata': {
                    'title': doc['title'],
                    'tags': doc['tags'],
                    'text': doc['text'][:200]
                }
            }
            
            # Update inverted index barrels
            for term_id, positions in document_terms.items():
                barrel_id = self._get_barrel_id(term_id, metadata)
                barrel_path = self._get_barrel_path(barrel_id)
                barrel = self._load_json(barrel_path)
                
                term_id_str = str(term_id)
                if term_id_str not in barrel:
                    print(f"Adding new term to barrel: {term_id_str}")
                    barrel[term_id_str] = {}
                
                # Create posting entry
                posting = self._create_posting_entry(positions, field_lengths)
                barrel[term_id_str][doc_id] = posting
                print(f"Added posting for term {term_id_str} in barrel {barrel_id}")
                
                self._save_json(barrel, barrel_path)
            
            # Save forward index batch
            self._save_json(forward_batch, self.path_manager.paths['forward_index'])
            
            return True, None
            
        except Exception as e:
            return False, f"Error adding document: {str(e)}"
        
        
if __name__ == "__main__":
    doc = {
        'doc_id': 1,
        'title': 'Computer networks testing is what I want to check here right now',
        'text': 'Python is a versatile programming language.',
        'tags': 'python, programming, language'
    }
    
    doc_adder = DocumentAdder()
    success, message = doc_adder.add_document(doc)
    if success:
        print("Document added successfully!")
    else:
        print(f"Failed to add document: {message}")