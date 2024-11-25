import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

# so i read the csv
articles_df =  pd.read_csv('medium_articles.csv')
          
# now I shall download the necessary nltk data
nltk.download('punkt')
nltk.download('stopwords')

stop_words = set(stopwords.words('english'))



# using a function named tokenize_column whihc would tokenize some specific columns like in muy case there is 
def tokenize_column(column):
    return column.apply(lambda x: [word.lower() for word in word_tokenize(x) if word.isalnum() and word.lower() not in stop_words])

articles_df["title_tokens"] = tokenize_column(articles_df["title"])
articles_df["text_tokens"] = tokenize_column(articles_df["text"])
