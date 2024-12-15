import pandas as pd
import os
from globalVariables import BARREL_SIZE
# loading the sorted inverted index
inverted_index_df = pd.read_csv('inverted_indexing_array.csv', encoding='utf-8')

# Making a folder for the barrels
os.makedirs("barrels", exist_ok=True)

# starting value for barrel id and start id
current_barrel_id = 1
current_start_id = 1

# I will get aa barrel range from here 
def get_barrel_id(word_id, barrel_size):
    return (word_id - 1) // barrel_size + 1

# I will get the start id of the barrel from here
for barrel_id, group in inverted_index_df.groupby(
    inverted_index_df['WordID'] // BARREL_SIZE
):
    # Save each barrel as a separate file
    barrel_filename = f"barrels/barrel_{barrel_id + 1}.csv"
    group.to_csv(barrel_filename, index=False, encoding='utf-8')
    print(f"Barrel {barrel_id + 1} saved to '{barrel_filename}'.")

print("All barrels have been created and saved.")
