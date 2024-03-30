from gensim.models import LdaModel
from gensim.corpora import Dictionary
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import nltk
import string
import os
import joblib
import re
import pickle
from nltk.stem import SnowballStemmer
from pathlib import Path


class LDAModel:
    def __init__(self):
        nltk.download('punkt')
        nltk.download('stopwords')
        script_dir = os.path.dirname(__file__)
        model_path = os.path.join(script_dir, 'models/lda_model.pkl')
        dictionary_path = os.path.join(script_dir, 'models/dictionary.pkl')

        self.lda_model = joblib.load(model_path)
        self.dictionary = joblib.load(dictionary_path)

    @staticmethod
    def preprocess_text(text):
        text = re.sub(r'[^a-zA-ZáéíóúüÁÉÍÓÚÜÑñ\s]', '', text)
        # Tokenize text
        tokens = word_tokenize(text)
        # Remove punctuation and convert to lowercase
        tokens = [token.lower() for token in tokens if token not in string.punctuation]
        # Remove stopwords
        stop_words = set(stopwords.words('spanish'))
        tokens = [token for token in tokens if token not in stop_words]
        # Stemming
        stemmer = SnowballStemmer('spanish')
        tokens = [stemmer.stem(token) for token in tokens]
        return tokens

    def predict_topic(self, text):
        tokens = self.preprocess_text(text)
        bow = self.dictionary.doc2bow(tokens)
        topic_distribution = self.lda_model.get_document_topics(bow)
        dominant_topic = max(topic_distribution, key=lambda x: x[1])[0]
        return dominant_topic
