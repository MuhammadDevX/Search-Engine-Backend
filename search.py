import json
import numpy as np
from typing import Dict, List, Tuple, Set
from collections import defaultdict
import Levenshtein
from nltk.corpus import wordnet
import heapq
from preprocessAndLexiconGen import LexiconLoader, TextPreprocessor
from forwardIndexGenerator import ForwardIndexLoader
from invertedIndexGenerator import InverseIndexLoader
import numpy as np
from typing import Dict, List, Tuple, Set
def vector_similarity(vec1: np.ndarray, vec2: np.ndarray) -> float:
    """
    Calculate similarity between two vectors using dot product and magnitude.
    This is essentially a manual implementation of cosine similarity.
    """
    # Calculate dot product
    dot_product = np.dot(vec1, vec2)
    
    # Calculate magnitudes
    magnitude1 = np.sqrt(np.sum(vec1 ** 2))
    magnitude2 = np.sqrt(np.sum(vec2 ** 2))
    
    # Avoid division by zero
    if magnitude1 == 0 or magnitude2 == 0:
        return 0.0
        
    # Calculate cosine similarity
    return dot_product / (magnitude1 * magnitude2)

class QueryProcessor:
    def __init__(self, lexicon_loader: LexiconLoader, text_preprocessor: TextPreprocessor):
        self.lexicon = lexicon_loader
        self.text_preprocessor = text_preprocessor
        self.word_embeddings = self._load_word_embeddings()
        
    def _load_word_embeddings(self) -> Dict[str, np.ndarray]:
        """Load or generate word embeddings for similarity comparison."""
        # In practice, you would load pre-trained embeddings
        # For this example, we'll create simple term frequency based embeddings
        embeddings = {}
        for word in self.lexicon.word_to_id.keys():
            stats = self.lexicon.get_word_stats(word)
            if stats:
                # Create a simple embedding based on term statistics
                embedding = np.array([
                    stats.get('doc_frequency', 0),
                    sum(stats.get('field_frequencies', {}).values()),
                    len(stats.get('document_occurrences', set()))
                ])
                embeddings[word] = embedding / np.linalg.norm(embedding)
        return embeddings
    
    def correct_spelling(self, word: str, max_distance: int = 2) -> str:
        """Correct spelling using Levenshtein distance."""
        if word in self.lexicon.word_to_id:
            return word
            
        candidates = []
        for dict_word in self.lexicon.word_to_id.keys():
            distance = Levenshtein.distance(word, dict_word)
            if distance <= max_distance:
                candidates.append((dict_word, distance))
        
        if candidates:
            return min(candidates, key=lambda x: x[1])[0]
        return word
    
    def find_similar_words(self, word: str, threshold: float = 0.7) -> List[str]:
        """Find similar words using custom vector similarity."""
        if word not in self.word_embeddings:
            return []
            
        word_vector = self.word_embeddings[word]
        similar_words = []
        
        for other_word, other_vector in self.word_embeddings.items():
            if other_word != word:
                # Use our custom similarity function instead of sklearn's cosine_similarity
                similarity = vector_similarity(word_vector, other_vector)
                if similarity >= threshold:
                    similar_words.append((other_word, similarity))
        
        # Return top 5 similar words
        return [word for word, _ in sorted(similar_words, key=lambda x: x[1], reverse=True)[:5]]
      
    def process_query(self, query: str) -> Tuple[List[str], Dict[str, List[str]]]:
        """Process query with spelling correction and similar words."""
        # Preprocess query
        processed_query = self.text_preprocessor.clean_and_lemmatize(query)
        query_terms = processed_query.split()
        
        # Correct spellings and find similar words
        corrected_terms = []
        similar_terms = {}
        
        for term in query_terms:
            # Correct spelling
            corrected_term = self.correct_spelling(term)
            corrected_terms.append(corrected_term)
            
            # Find similar words
            similar_terms[corrected_term] = self.find_similar_words(corrected_term)
        
        return corrected_terms, similar_terms

# class QueryProcessor:
#     def __init__(self, lexicon_loader: LexiconLoader, text_preprocessor: TextPreprocessor):
#         self.lexicon = lexicon_loader
#         self.text_preprocessor = text_preprocessor
#         self.word_embeddings = self._load_word_embeddings()
        
#     def _load_word_embeddings(self) -> Dict[str, np.ndarray]:
#         """Load or generate word embeddings for similarity comparison."""
#         # In practice, you would load pre-trained embeddings
#         # For this example, we'll create simple term frequency based embeddings
#         embeddings = {}
#         for word in self.lexicon.word_to_id.keys():
#             stats = self.lexicon.get_word_stats(word)
#             if stats:
#                 # Create a simple embedding based on term statistics
#                 embedding = np.array([
#                     stats.get('doc_frequency', 0),
#                     sum(stats.get('field_frequencies', {}).values()),
#                     len(stats.get('document_occurrences', set()))
#                 ])
#                 embeddings[word] = embedding / np.linalg.norm(embedding)
#         return embeddings
    
#     def correct_spelling(self, word: str, max_distance: int = 2) -> str:
#         """Correct spelling using Levenshtein distance."""
#         if word in self.lexicon.word_to_id:
#             return word
            
#         candidates = []
#         for dict_word in self.lexicon.word_to_id.keys():
#             distance = Levenshtein.distance(word, dict_word)
#             if distance <= max_distance:
#                 candidates.append((dict_word, distance))
        
#         if candidates:
#             return min(candidates, key=lambda x: x[1])[0]
#         return word
    
#     def find_similar_words(self, word: str, threshold: float = 0.7) -> List[str]:
#         """Find similar words using cosine similarity."""
#         if word not in self.word_embeddings:
#             return []
            
#         word_vector = self.word_embeddings[word]
#         similar_words = []
        
#         for other_word, other_vector in self.word_embeddings.items():
#             if other_word != word:
#                 similarity = cosine_similarity(
#                     word_vector.reshape(1, -1),
#                     other_vector.reshape(1, -1)
#                 )[0][0]
#                 if similarity >= threshold:
#                     similar_words.append((other_word, similarity))
        
#         # Return top 5 similar words
#         return [word for word, _ in sorted(similar_words, key=lambda x: x[1], reverse=True)[:5]]
    
#     def process_query(self, query: str) -> Tuple[List[str], Dict[str, List[str]]]:
#         """Process query with spelling correction and similar words."""
#         # Preprocess query
#         processed_query = self.text_preprocessor.clean_and_lemmatize(query)
#         query_terms = processed_query.split()
        
#         # Correct spellings and find similar words
#         corrected_terms = []
#         similar_terms = {}
        
#         for term in query_terms:
#             # Correct spelling
#             corrected_term = self.correct_spelling(term)
#             corrected_terms.append(corrected_term)
            
#             # Find similar words
#             similar_terms[corrected_term] = self.find_similar_words(corrected_term)
        
#         return corrected_terms, similar_terms

class SearchEngine:
    def __init__(self, 
                 inverse_index_loader: InverseIndexLoader,
                 forward_index_loader: ForwardIndexLoader,
                 query_processor: QueryProcessor):
        self.inverse_index = inverse_index_loader
        self.forward_index = forward_index_loader
        self.query_processor = query_processor
        
    def _calculate_bm25_score(self, 
                             term: str, 
                             doc_id: str,
                             field_weights: Dict[str, float] = None) -> float:
        """Calculate BM25 score for a term-document pair."""
        if field_weights is None:
            field_weights = {'title': 3.0, 'text': 1.0, 'tags': 2.0}
            
        k1 = 1.5
        b = 0.75
        
        # Get document length and average document length
        doc_length = self.forward_index.get_document_size(doc_id)
        avg_doc_length = self.forward_index.get_average_document_size()
        
        # Get term frequencies in different fields
        field_freqs = self.inverse_index.get_field_frequencies(term, doc_id)
        
        # Calculate weighted term frequency
        weighted_tf = sum(
            freq * field_weights.get(field, 1.0)
            for field, freq in field_freqs.items()
        )
        
        # Get IDF score
        idf = self.inverse_index.get_idf(term)
        
        # Calculate BM25 score
        numerator = weighted_tf * (k1 + 1)
        denominator = weighted_tf + k1 * (1 - b + b * doc_length / avg_doc_length)
        
        return idf * numerator / denominator
    
    def _calculate_proximity_score(self, 
                                 terms: List[str], 
                                 doc_id: str,
                                 window_size: int = 10) -> float:
        """Calculate proximity score for multiple terms."""
        positions = [
            self.inverse_index.get_positions(term, doc_id)
            for term in terms
        ]
        
        if not all(positions):
            return 0.0
            
        # Find minimum distance between all terms
        min_distance = float('inf')
        for positions_combination in zip(*positions):
            max_pos = max(positions_combination)
            min_pos = min(positions_combination)
            distance = max_pos - min_pos
            if distance < min_distance:
                min_distance = distance
        
        # Convert distance to score (closer terms get higher scores)
        if min_distance >= window_size:
            return 0.0
        return 1.0 - (min_distance / window_size)
    
    def search(self, 
              query: str, 
              max_results: int = 10,
              window_size: int = 10) -> List[Dict]:
        """
        Search for documents matching the query.
        Returns list of documents with titles, texts, and URLs.
        """
        # Process query
        corrected_terms, similar_terms = self.query_processor.process_query(query)
        
        # Collect all terms to search (original + similar)
        all_terms = set(corrected_terms)
        for term_similars in similar_terms.values():
            all_terms.update(term_similars)
        
        # Calculate scores for each document
        doc_scores = defaultdict(float)
        
        # Score documents containing query terms
        for term in all_terms:
            term_data = self.inverse_index.get_term_data(term)
            for doc_id in term_data.get('document_frequencies', {}):
                # Calculate BM25 score
                bm25_score = self._calculate_bm25_score(term, doc_id)
                
                # Add similarity factor if term is similar (not exact)
                similarity_factor = 1.0
                if term not in corrected_terms:
                    for original_term, similar_list in similar_terms.items():
                        if term in similar_list:
                            # Reduce score for similar terms
                            similarity_factor = 0.5
                            break
                
                doc_scores[doc_id] += bm25_score * similarity_factor
        
        # Add proximity bonus for multi-term queries
        if len(corrected_terms) > 1:
            for doc_id in doc_scores:
                proximity_score = self._calculate_proximity_score(
                    corrected_terms, doc_id, window_size
                )
                doc_scores[doc_id] *= (1 + proximity_score)
        
        # Get top documents
        top_docs = heapq.nlargest(
            max_results,
            doc_scores.items(),
            key=lambda x: x[1]
        )
        
        # Fetch document contents
        results = []
        for doc_id, score in top_docs:
            doc_metadata = self.forward_index.get_document_metadata(doc_id)
            results.append({
                'doc_id': doc_id,
                'title': doc_metadata.get('title', ''),
                'text': doc_metadata.get('text', ''),
                'url': doc_metadata.get('url', ''),
                'score': score,
                'matched_terms': [
                    term for term in all_terms
                    if self.inverse_index.get_document_frequency(term, doc_id) > 0
                ]
            })
        
        return results

def main():
    """Example usage of the search engine."""
    # Initialize components
    lexicon_loader = LexiconLoader("lexicon_output")
    text_preprocessor = TextPreprocessor()
    inverse_index_loader = InverseIndexLoader("inverse_index_output", lexicon_loader)
    forward_index_loader = ForwardIndexLoader("forward_index_output", lexicon_loader)
    
    # Initialize query processor and search engine
    query_processor = QueryProcessor(lexicon_loader, text_preprocessor)
    search_engine = SearchEngine(
        inverse_index_loader,
        forward_index_loader,
        query_processor
    )
    
    # Example search
    query = "learning"
    results = search_engine.search(query, max_results=5)
    
    # Print results
    print(f"Search results for: {query}\n")
    for i, result in enumerate(results, 1):
        print(f"Result {i}:")
        print(f"Title: {result['title']}")
        print(f"URL: {result['url']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Matched terms: {', '.join(result['matched_terms'])}")
        print(f"Text preview: {result['text'][:200]}...\n")

if __name__ == "__main__":
    main()