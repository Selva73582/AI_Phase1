import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.preprocessing import LabelEncoder


true_data = pd.read_csv("True.csv")
fake_data = pd.read_csv("Fake.csv")


true_data['label'] = 1
fake_data['label'] = 0


def preprocess_text(text):
    stop_words = set(stopwords.words('english'))
    tokens = word_tokenize(text.lower())
    tokens = [word for word in tokens if word.isalpha()]
    tokens = [word for word in tokens if word not in stop_words]
    return ' '.join(tokens)



data['PreprocessedText'] = data['text'].apply(preprocess_text)

data['text'] = data['text'].apply(preprocess_text)
