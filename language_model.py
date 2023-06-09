#!/usr/bin/env python
# coding: utf-8

import pandas as pd
import os
import matplotlib.pyplot as plt 
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
path="data/"
os.chdir(path)


languages= ' Spanish,English,Dutch,French,German,Portuguese,Spanish, Spanish, Croatian,Danish,English,Arabic,English,French,German,Icelandic,Persian,Turkic,Japanese,  Spanish,French,English,Spanish,Spanish,Polish,Russian,Arabic,French,Serbian,Korean,Spanish,Swedish,German,Italian,Arabic,Spanish'











languages_list=list(languages.strip().split(','))
plt.bar(pd.Series(languages_list).value_counts().index,pd.Series(languages_list).value_counts().values,color='skyblue')
plt.xticks(rotation=90)
plt.title('languages spoken by countries')




from transformers import pipeline

# Load the pre-trained BERT model for translation
translator = pipeline("translation_en_to_es", model="Helsinki-NLP/opus-mt-en-es")

# Translate a Spanish text to English
spanish_text = "Hola, ¿cómo estás?"
english_text = translator(spanish_text, max_length=40)[0]['translation_text']

print(english_text)
# Output: "Hi, how are you?"


