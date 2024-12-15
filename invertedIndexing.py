import pandas as pd
from collections import defaultdict, Counter
import ast 

# Load the forward index
forward_index_df = pd.read_csv(r'forward_indexing.csv', encoding='utf-8')

# Initialize the inverted index
# Structure: { word_id: [{doc_id: freq}, ...], ... }
inverted_index = defaultdict(list)

# Build the inverted index with frequency counts
for _, row in forward_index_df.iterrows():
    doc_id = str(row['docID'])
    word_ids = ast.literal_eval(row['WordIDs'])  # Safely parse the list of WordIDs
    word_freq = Counter(word_ids)  # Count the frequency of each word_id in the current document
    
    for word_id, freq in word_freq.items():
        inverted_index[word_id].append({'DocID': doc_id, 'Frequency': freq})

# Convert the inverted index to a list of dictionaries for saving
inverted_index_records = [{'WordID': word_id, 'Postings': inverted_index[word_id]} 
                          for word_id in inverted_index]

# Convert the list to a DataFrame
inverted_index_df = pd.DataFrame(inverted_index_records)

# Save the inverted index to CSV in array form
inverted_index_df.to_csv(r'inverted_indexing_array.csv', index=False, encoding='utf-8')
print("Inverted index with array postings saved to 'inverted_indexing.csv'")
