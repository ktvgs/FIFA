import pandas as pd
import nltk
nltk.download('punkt')
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer

country=[]
dataset='England_tweets_of_interest.csv'
with open(dataset,encoding="utf-8") as file:
    
    for line in file:
        country.append(line.split('|@|||$|')[2])
df=pd.DataFrame(country,columns=['text'])

def preprocess_text(text):
    # Tokenize the text into individual words
    tokens = word_tokenize(text)
    
    # Remove non-alphabetic characters and convert to lowercase
    clean_tokens = [token.lower() for token in tokens if token.isalpha()]
    
    # Remove stop words
    stop_words = set(stopwords.words('english'))
    filtered_tokens = [token for token in clean_tokens if not token in stop_words]
    
    # Stem the remaining words using Porter stemmer
    ps = PorterStemmer()
    stemmed_tokens = [ps.stem(token) for token in filtered_tokens]
    
    # Join the stemmed tokens back into a single string
    processed_text = ' '.join(stemmed_tokens)
    
    return processed_text

# Apply the preprocessing function to the text column
df['processed_text'] = df['text'].apply(preprocess_text)

# Filter out non-English words using NLTK's English word corpus
english_words = set(nltk.corpus.words.words())
df['english_only_text'] = df['processed_text'].apply(lambda x: ' '.join(w for w in x.split() if w in english_words))

text_values1 = df['english_only_text'].dropna().tolist()
text_values2 = df['processed_text'].dropna().tolist()

def domain(text_values):
    tokens = [word.lower() for text in text_values for word in nltk.word_tokenize(text)]

    # Calculate the frequency of each word in the text corpus
    freq_dist = nltk.FreqDist(tokens)

    # Identify common words that occur frequently but do not carry much meaning
    common_words = [word for word, count in freq_dist.most_common(200) if count > 50]

    # Remove stopwords that are already included in a general-purpose stopwords list
    stopwords = set(nltk.corpus.stopwords.words('english'))
    domain_stopwords = set(common_words) - stopwords
    return domain_stopwords

l1=list(domain(text_values1))
l2=list(domain(text_values2))

pd.DataFrame(l1).to_csv('english_only.csv')
pd.DataFrame(l2).to_csv('preprocessed.csv')




