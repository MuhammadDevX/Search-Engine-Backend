import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from typing import Dict, Set, List
import json
import os
from collections import defaultdict

class TextPreprocessor:
    def __init__(self):
        self.stop_words = set(stopwords.words("english"))
        self.lemmatizer = WordNetLemmatizer()
        
    def clean_characters(self, text: str) -> str:
        """Remove special characters and convert to lowercase."""
        text = re.sub(r'[^\w\s]','', str(text))
        return text.lower()
    
    def clean_and_lemmatize(self, text: str) -> str:
        """Clean text, remove stopwords, and lemmatize."""
        if not isinstance(text, str):
            return ""
        tokens = word_tokenize(text.lower())
        clean_list = [self.lemmatizer.lemmatize(word) 
                     for word in tokens 
                     if word.isalpha() and word not in self.stop_words]
        return " ".join(clean_list)
    
    def process_tags(self, tags_text: str) -> str:
        """Process tags from string representation to space-separated text."""
        try:
            tags = tags_text[1:-1].split(", ")
            tags = [w[1:-1] for w in tags]
            return " ".join(tags)
        except Exception:
            return ""

class DocumentProcessor:
    def __init__(self):
        self.text_preprocessor = TextPreprocessor()
        
    def process_document(self, doc: pd.Series) -> Dict:
        """Process a single document's fields separately."""
        # Process each field
        title_processed = self.text_preprocessor.clean_and_lemmatize(
            self.text_preprocessor.clean_characters(doc.get('title', ''))
        )
        
        text_processed = self.text_preprocessor.clean_and_lemmatize(
            self.text_preprocessor.clean_characters(doc.get('text', ''))
        )
        
        tags_text = self.text_preprocessor.process_tags(doc.get('tags', '[]'))
        tags_processed = self.text_preprocessor.clean_and_lemmatize(tags_text)
        
        return {
            'doc_id': doc.name,
            'title': title_processed,
            'text': text_processed,
            'tags': tags_processed
        }
    
    def process_dataset(self, input_file: str, output_file: str):
        """Process the entire dataset and save as JSON."""
        # Read and clean dataset
        df = pd.read_csv(input_file)
        df = df.drop_duplicates().dropna()
        
        # Process each document
        processed_docs = [
            self.process_document(row) 
            for _, row in df.iterrows()
        ]
        
        # Save as JSON for faster retrieval
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump({
                'documents': processed_docs,
                'metadata': {
                    'total_documents': len(processed_docs),
                    'fields': ['title', 'text', 'tags']
                }
            }, f, indent=2)
        
        return processed_docs

class LexiconGenerator:
    def __init__(self):
        self.word_to_id = {}
        self.word_stats = defaultdict(lambda: {
            'doc_frequency': 0,
            'field_frequencies': defaultdict(int),
            'document_occurrences': set()
        })
        self.next_word_id = 0
        
    def _get_or_create_word_id(self, word: str) -> int:
        """Get existing word ID or create new one."""
        if word not in self.word_to_id:
            self.word_to_id[word] = self.next_word_id
            self.next_word_id += 1
        return self.word_to_id[word]
    
    def process_document(self, doc: Dict):
        """Process a single document and update statistics."""
        doc_id = doc['doc_id']
        
        # Process each field
        for field in ['title', 'text', 'tags']:
            words = set(doc[field].split())  # Using set for unique words
            for word in words:
                word_id = self._get_or_create_word_id(word)
                self.word_stats[word]['field_frequencies'][field] += 1
                self.word_stats[word]['document_occurrences'].add(doc_id)
    
    def generate_lexicon(self, processed_docs: List[Dict], output_dir: str):
        """Generate lexicon and statistics in JSON format."""
        os.makedirs(output_dir, exist_ok=True)
        
        # Process all documents
        for doc in processed_docs:
            self.process_document(doc)
        
        # Prepare lexicon data
        lexicon_data = {
            'word_to_id': self.word_to_id,
            'id_to_word': {str(v): k for k, v in self.word_to_id.items()},
            'metadata': {
                'total_words': len(self.word_to_id),
                'max_word_id': self.next_word_id - 1
            }
        }
        
        # Prepare statistics data
        stats_data = {
            word: {
                'word_id': self.word_to_id[word],
                'doc_frequency': len(stats['document_occurrences']),
                'field_frequencies': dict(stats['field_frequencies'])
            }
            for word, stats in self.word_stats.items()
        }
        
        # Save lexicon files
        with open(f"{output_dir}/lexicon.json", 'w', encoding='utf-8') as f:
            json.dump(lexicon_data, f, indent=2)
            
        with open(f"{output_dir}/word_statistics.json", 'w', encoding='utf-8') as f:
            json.dump(stats_data, f, indent=2)
        
        return lexicon_data, stats_data

class LexiconLoader:
    def __init__(self, lexicon_dir: str):
        """Load lexicon and statistics from JSON files."""
        with open(f"{lexicon_dir}/lexicon.json", 'r', encoding='utf-8') as f:
            lexicon_data = json.load(f)
            self.word_to_id = lexicon_data['word_to_id']
            self.id_to_word = lexicon_data['id_to_word']
            self.metadata = lexicon_data['metadata']
        
        with open(f"{lexicon_dir}/word_statistics.json", 'r', encoding='utf-8') as f:
            self.word_stats = json.load(f)
    
    def get_word_id(self, word: str) -> int:
        """Get word ID for a given word."""
        return self.word_to_id.get(word, -1)
    
    def get_word(self, word_id: int) -> str:
        """Get word for a given word ID."""
        return self.id_to_word.get(str(word_id), "")
    
    def get_word_stats(self, word: str) -> Dict:
        """Get statistics for a given word."""
        return self.word_stats.get(word, {})

# Example usage
if __name__ == "__main__":
    # Process documents
    doc_processor = DocumentProcessor()
    processed_docs = doc_processor.process_dataset("test.csv", "processed_articles.json")
    
    # Generate lexicon
    lexicon_generator = LexiconGenerator()
    lexicon_data, stats_data = lexicon_generator.generate_lexicon(processed_docs, "lexicon_output")
    
    # Load and use lexicon
    lexicon = LexiconLoader("lexicon_output")
    
    # Example lookups
    word = "example"
    word_id = lexicon.get_word_id(word)
    stats = lexicon.get_word_stats(word)
    print(word,stats)