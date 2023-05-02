import pandas as pd 
import re
import string
import nltk
import re


from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import words
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from scipy.stats import pearsonr
from sklearn.feature_extraction.text import CountVectorizer
from nltk.sentiment import SentimentIntensityAnalyzer
import matplotlib.ticker as ticker
import matplotlib.pyplot as plt
import scipy as sp

df1=pd.read_csv('australia_pp.csv')
df2=pd.read_csv('nigeria_pp.csv')

df1=df1.drop(columns=['Unnamed: 0'],axis=1)
df2=df2.drop(columns=['Unnamed: 0'],axis=1)

df1.dropna(inplace=True)
df2.dropna(inplace=True)

domain_stopwords=['worldcup','win','defeat','road', 'green', 'need', 'save', 'time', 'chang', 'get', 'took', 'sport', 'drink', 'less', 'trump',
 'lo', 'load', 'forward', 'phone', 'come', 'music', 'plan', 'week', 'hit', 'kick', 'safe', 'state', 'year', 'import', 
 'food', 'fall', 'x', 'book', 'stay', 'poor', 'la', 'case', 'cover', 'love', 'wind', 'dog', 'number', 'min', 'go', 'world', 'game', 'join', 'un', 'along', 'shop', 'still', 'beat', 'fight', 'forget', 'project', 'en', 'wrong', 'work', 'session', 'use', 'jack', 'three', 'also', 'later', 'via', 'fair', 'pick', 'ticket', 'weekend', 'u', 'say', 'wed', 'night', 'hour', 'soon', 'avail', 'made', 'woman', 'almost', 'tonight', 'long', 'problem', 'sick', 'fantast', 'twitter', 'remain', 'winner', 'mum', 'watch', 'free', 'school', 'one', 'morn', 'score', 'enjoy', 'ca', 'da', 'bless', 'cross', 'media', 'el', 'posit', 'keep', 'heart', 'laugh', 'win', 'mani', 'eye', 'girl', 'like', 'move', 'album', 'side', 'walk', 'present', 'sure', 'visit', 'market', 'dad', 'beer', 'nation', 'joke', 'custom', 'luck', 'n', 'date', 'bed', 'act', 'want', 'hate', 'class', 'stage', 'vote', 'rather', 'holiday', 'found', 'ask', 'player', 'quit', 'gone', 'lot', 'actual', 'hire', 'season', 'stop', 'told', 'govern', 'ever', 'person', 'car', 'dream', 'heat', 'pay', 'idea', 'eu', 'tri', 'spend', 'answer', 'good', 'creat', 'due', 'kingdom', 'around', 'tour', 'sell', 'went', 'came', 'turn', 'interest', 'seen', 'liter', 'sort', 'summer', 'rise', 'next', 'ago', 'yeah', 'across', 'instead', 'worst', 'ya', 'word', 'humid', 'set', 'race', 'view', 'level', 'sleep', 'lost', 'understand', 'birthday', 'hair', 'happen', 'never', 'whole', 'agre', 'fit', 'ha', 'speak', 'success', 'back', 'big', 'gon', 'top', 'either', 'cloud', 'hold', 'c', 'video', 'design', 'enough', 'light', 'meet', 'alway', 'call', 'young', 'could', 'total', 'super', 'step', 'current', 'eat', 'taken', 'link', 'god', 'travel', 'result', 'thought', 'lie', 'perfect', 'old', 'support', 'base', 'talk', 'tell', 'late', 'think', 'finish', 'today', 'give', 'bu', 'rest', 'franc', 'hell', 'would', 'deal', 'short', 'past', 'account', 'share', 'fan', 'high', 'afternoon', 'job', 'cut', 'train', 'friend', 'miss', 'far', 'man', 'find', 'weather', 'public', 'card', 'bet', 'final', 'bar', 'sound', 'least', 'said', 'moment', 'cup', 'close', 'sale', 'pass', 'black', 'follow', 'hey', 'record', 'kind', 'match', 'water', 'rain', 'trip', 'month', 'thing', 'art', 'recommend', 'expect', 'better', 'money', 'done', 'life', 'full', 'stuff', 'proper', 'learn', 'best', 'lose', 'check', 'fun', 'wear', 'fire', 'white', 'mean', 'play', 'true', 'bring', 'round', 'song', 'half', 'june', 'brilliant', 'question', 'staff', 'club', 'drive', 'first', 'sun', 'pride', 'queen', 'let', 'v', 'latest', 'offer', 'ye', 'news', 'age', 'left', 'take', 'garden', 'home', 'wonder', 'worth', 'touch', 'allow', 'hard', 'se', 'open', 'without', 'member', 'aw', 'write', 'guess', 'yesterday', 'pic', 'thank', 'south', 'feel', 'right', 'beauti', 'hi', 'sad', 'award', 'care', 'team', 'air', 'though', 'health', 'well', 'send', 'sit', 'may', 'listen', 'photo', 'lad', 'ball', 'respect', 'glad', 'bit', 'greater', 'real', 'mind', 'social', 'room', 'much', 'mate', 'fact', 'buy', 'new', 'west', 'great', 'order', 'show', 'sign', 'els', 'film', 'click', 'read', 'yet', 'report', 'run', 'hello', 'unit', 'two', 'red', 'wait', 'put', 'comment', 'local', 'law', 'stand', 'last', 'head', 'small', 'see', 'b', 'pop', 'nice', 'reason', 'build', 'end', 'e', 'student', 'box', 'day', 'mad', 'press', 'die', 'russia', 'make', 'tomorrow', 'station', 'catch', 'look', 'treat', 'guy', 'human', 'behind', 'goal', 'spot', 'drop', 'power', 'tweet', 'point', 'know', 'island', 'dead', 'cool', 'kill', 'lead', 'bad', 'line', 'hot', 'shot', 'cheer', 'hear', 'away', 'hope', 'face', 'break', 'place', 'space', 'north', 'hand', 'na', 'seem', 'oh', 'wan', 'boy', 'matter', 'men', 'name', 'product', 'us', 'probabl', 'discuss', 'front', 'way', 'given', 'shame', 'excel', 'street', 'detail', 'colour', 'might', 'differ', 'got', 'wish',
 'special', 'saw', 'must', 'huge', 'help', 'list', 'wow', 'son', 'start', 'mine', 'perform', 'even', 'de', 'labour', 'group', 'live', 'post', 'part', 'blue', 'park', 'second', 'event', 'fine']

tfidf_vectorizer = TfidfVectorizer(stop_words=domain_stopwords)
tfidf1 = tfidf_vectorizer.fit_transform(df1['english_only_text'])
tfidf2 = tfidf_vectorizer.transform(df2['english_only_text'])

cosine_similarities = cosine_similarity(tfidf1, tfidf2)
avg_cosine_similarity = np.mean(cosine_similarities)
print(avg_cosine_similarity)