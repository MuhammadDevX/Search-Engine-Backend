import pandas as pd

# Read the CSV file
df = pd.read_csv('medium_articles.csv')

# Extract the first 100 rows (or adjust based on your criteria)
df_subset = df.head(100)

# Save the subset to a new CSV file
df_subset.to_csv('test.csv', index=False)

print("100 articles have been saved to 'test.csv'")
