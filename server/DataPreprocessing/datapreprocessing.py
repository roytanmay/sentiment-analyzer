import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.stem import WordNetLemmatizer
from sklearn.base import BaseEstimator, TransformerMixin
import re

stop_words = stopwords.words('english')
new_stopwords = ["would", "shall", "could", "might", "film", "movie", "director", "scene", "character", "actor", "actress", "morio", "la", "blah", "monday", "friday", "saturday", "sunday", "morning", "evening"]
stop_words.extend(new_stopwords)

negations_and_sentiment_words = ["not", "no", "never", "n't", "none", "good", "bad", "love", "hate"]
for word in negations_and_sentiment_words:
    if word in stop_words:
        stop_words.remove(word)

stop_words = set(stop_words)
# print(stop_words)


#Removing special character
def remove_special_character(content):
    # return re.sub(r'\W+',' ', content )
    return re.sub(r'\[[^&@#!]]*\]', '', content)

# Removing URL's
def remove_url(content):
    return re.sub(r'http\S+', '', content)

#Removing the stopwords from text
def remove_stopwords(content):
    clean_data = []
    for i in content.split():
        if i.strip().lower() not in stop_words and i.strip().lower().isalpha():
            clean_data.append(i.strip().lower())
    return " ".join(clean_data)

# Expansion of english contractions
def contraction_expansion(content):
    content = re.sub(r"won\'t", "would not", content)
    content = re.sub(r"can\'t", "can not", content)
    content = re.sub(r"don\'t", "do not", content)
    content = re.sub(r"shouldn\'t", "should not", content)
    content = re.sub(r"needn\'t", "need not", content)
    content = re.sub(r"hasn\'t", "has not", content)
    content = re.sub(r"haven\'t", "have not", content)
    content = re.sub(r"weren\'t", "were not", content)
    content = re.sub(r"mightn\'t", "might not", content)
    content = re.sub(r"didn\'t", "did not", content)
    content = re.sub(r"n\'t", " not", content)
    return content

#Data preprocessing
def data_cleaning(content):
    content = contraction_expansion(content)
    content = remove_special_character(content)
    content = remove_url(content)
    
    content = remove_stopwords(content)    
    return content


class DataCleaning(BaseEstimator, TransformerMixin):
    def __init__(self):
        print("Calling __init__ ...")

    def fit(self, X, y=None):
        print("Calling fit ...")
        return self

    def transform(self, X, y=None):
        print("Calling transform")
        X = X.apply(data_cleaning)
        return X
    
class LemmaTokenizer(object):
    def __init__(self):
        self.wordnetlemma = WordNetLemmatizer()

    def __call__(self, reviews):
        return [self.wordnetlemma.lemmatize(word) for word in word_tokenize(reviews)]