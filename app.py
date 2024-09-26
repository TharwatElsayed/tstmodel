# Import required libraries
import streamlit as st
import re
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.utils import pad_sequences
import pickle

# Preprocessing functions
space_pattern = '\s+'
giant_url_regex = ('http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|'
        '[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
mention_regex = '@[\w\-]+'
emoji_regex = '&#[0-9]{4,6};'

def preprocess(text_string):
    parsed_text = re.sub(space_pattern, ' ', text_string)
    parsed_text = re.sub(giant_url_regex, '', parsed_text)
    parsed_text = re.sub(mention_regex, '', parsed_text)
    parsed_text = re.sub('RT', '', parsed_text)
    parsed_text = re.sub(emoji_regex, '', parsed_text)
    parsed_text = re.sub('…', '', parsed_text)
    return parsed_text

def preprocess_clean(text_string, remove_hashtags=True, remove_special_chars=True):
    text_string = preprocess(text_string)
    parsed_text = text_string.lower()
    parsed_text = re.sub('\'', '', parsed_text)
    parsed_text = re.sub(':', '', parsed_text)
    parsed_text = re.sub(',', '', parsed_text)
    parsed_text = re.sub('&amp', '', parsed_text)

    if remove_hashtags:
        parsed_text = re.sub('#[\w\-]+', '', parsed_text)
    if remove_special_chars:
        parsed_text = re.sub('(\!|\?)+', '', parsed_text)
    return parsed_text

def strip_hashtags(text):
    text = preprocess_clean(text, False, True)
    hashtags = re.findall('#[\w\-]+', text)
    for tag in hashtags:
        cleantag = tag[1:]
        text = re.sub(tag, cleantag, text)
    return text

# Stemming function
stemmer = PorterStemmer()
def stemming(text):
    stemmed_tweets = [stemmer.stem(t) for t in text.split()]
    return stemmed_tweets

# Streamlit application
st.title("Tweet Sentiment/Class Prediction")

# Input box for entering the tweet
user_input = st.text_area("Enter the tweet:", "I love this!")

# Button to trigger prediction
if st.button('Predict'):
    # Preprocessing steps
    preprocessed_tweet = preprocess(user_input)
    clean_tweet = preprocess_clean(preprocessed_tweet)
    stripped_tweet = strip_hashtags(clean_tweet)
    stemmed_tweet = stemming(stripped_tweet)
    
    # Tokenize and pad the tweet
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(stemmed_tweet)
    encoded_docs = tokenizer.texts_to_sequences(stemmed_tweet)
    encoded_docs = [item for sublist in encoded_docs for item in sublist]
    max_length = 100
    padded_docs = pad_sequences([encoded_docs], maxlen=max_length, padding='post')
    
    # Load the pre-trained model (ensure LR_model.pkl is in the same directory)
    with open('LR_model.pkl', 'rb') as f:
        LR_model = pickle.load(f)

    # Predict sentiment/class
    y_pred = LR_model.predict(padded_docs)

    # Map the prediction to a human-readable label
    label_map = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
    # Display prediction result
    st.write(f"Prediction: {label_map[y_pred[0]]}")
    st.write(f"Prediction: {y_pred}")
