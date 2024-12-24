import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from datetime import datetime
from typing import List, Dict, Optional
import numpy as np

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
    def __init__(self, preprocessor: TextPreprocessor):
        self.preprocessor = preprocessor
    
    def calculate_text_stats(self, text: str) -> Dict:
        """Calculate various text statistics for ranking."""
        words = text.split()
        return {
            'word_count': len(words),
            'unique_word_count': len(set(words)),
            'avg_word_length': np.mean([len(w) for w in words]) if words else 0
        }
    
    def extract_metadata(self, row: pd.Series) -> Dict:
        """Extract and process metadata from document."""
        metadata = {}
        
        # Process timestamp if available
        if 'timestamp' in row:
            try:
                metadata['timestamp'] = pd.to_datetime(row['timestamp'])
                metadata['age_days'] = (datetime.now() - metadata['timestamp']).days
            except:
                metadata['timestamp'] = None
                metadata['age_days'] = None
        
        # Process author information if available
        if 'author' in row:
            metadata['author'] = str(row['author'])
            
        # Process view count if available
        if 'views' in row:
            try:
                metadata['views'] = int(row['views'])
            except:
                metadata['views'] = 0
                
        # process url if available
        if 'url' in row:
            metadata['url'] = str(row['url'])
        return metadata

    def process_document(self, row: pd.Series) -> Dict:
        """Process a single document row."""
        # Process each field separately
        title_cleaned = self.preprocessor.clean_characters(row.get('title', ''))
        title_processed = self.preprocessor.clean_and_lemmatize(title_cleaned)
        
        text_cleaned = self.preprocessor.clean_characters(row.get('text', ''))
        text_processed = self.preprocessor.clean_and_lemmatize(text_cleaned)
        
        tags_text = self.preprocessor.process_tags(row.get('tags', '[]'))
        tags_processed = self.preprocessor.clean_and_lemmatize(tags_text)
        
        # Calculate statistics for each field
        title_stats = self.calculate_text_stats(title_processed)
        text_stats = self.calculate_text_stats(text_processed)
        tags_stats = self.calculate_text_stats(tags_processed)
        
        # Extract metadata
        metadata = self.extract_metadata(row)
        
        return {
            'title': title_processed,
            'title_length': title_stats['word_count'],
            'title_unique_words': title_stats['unique_word_count'],
            
            'text': text_processed,
            'text_length': text_stats['word_count'],
            'text_unique_words': text_stats['unique_word_count'],
            'text_avg_word_length': text_stats['avg_word_length'],
            
            'tags': tags_processed,
            'tag_count': tags_stats['word_count'],
            
            **metadata  # Include all metadata fields
        }

class SearchEnginePreprocessor:
    def __init__(self):
        self.text_preprocessor = TextPreprocessor()
        self.doc_processor = DocumentProcessor(self.text_preprocessor)
    
    def process_dataset(self, filepath: str, output_path: str):
        """Process the entire dataset and save results."""
        # Read and clean dataset
        df = pd.read_csv(filepath)
        df = df.drop_duplicates().dropna()
        
        # Process each document
        processed_docs = []
        for _, row in df.iterrows():
            processed_doc = self.doc_processor.process_document(row)
            processed_docs.append(processed_doc)
        
        # Create processed dataframe
        processed_df = pd.DataFrame(processed_docs)
        
        # Calculate additional ranking features
        if 'views' in processed_df.columns and 'age_days' in processed_df.columns:
            processed_df['popularity_score'] = processed_df['views'] / (processed_df['age_days'] + 1)
        
        # Save processed dataset
        processed_df.to_csv(output_path, index=False)
        return processed_df

# Example usage
if __name__ == "__main__":
    preprocessor = SearchEnginePreprocessor()
    processed_df = preprocessor.process_dataset("test.csv", "processed_articles.csv")