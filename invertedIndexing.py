import pandas as pd
from collections import defaultdict
import ast 

# Load the forward index
forward_index_df = pd.read_csv(r'forward_indexing.csv', encoding='utf-8')

# Initialize the inverted index
# Structure: { word_id: [{'DocID': doc_id, 'Positions': [pos1, pos2, ...], 'HitType': [type1, type2, ...]}], ... }

inverted_index = defaultdict(list)

# Build the inverted index with frequency counts
for _, row in forward_index_df.iterrows():
    doc_id = str(row['docID'])
    word_occurrences = ast.literal_eval(row['WordOccurrences'])  
    
    word_position = defaultdict(list)
    word_hit_type = defaultdict(list)
    
    for word_id,position, hit_type in word_occurrences:
        word_position[word_id].append(position)
        word_hit_type[word_id].append(hit_type)
        
    for word_id, positions in word_position.items():
        posting = {
            'DocID': doc_id,
            'Positions': positions,
            'HitType': word_hit_type[word_id]
        }
        inverted_index[word_id].append(posting)
    
inverted_index_records = [{'WordID': word_id, 'Postings': inverted_index[word_id]}
                                for word_id in inverted_index]

    
inverted_index_df = pd.DataFrame(inverted_index_records)
inverted_index_df = inverted_index_df.sort_values(by='WordID')



# Save the inverted index to CSV in array form
inverted_index_df.to_csv(r'inverted_indexing_array.csv', index=False, encoding='utf-8')
print("Inverted index with proximity and hit types saved to 'inverted_indexing.csv'")
