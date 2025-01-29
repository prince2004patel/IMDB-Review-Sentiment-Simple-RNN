import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import imdb
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import sequence
import streamlit as st

# load the IMDB dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value: key for key, value in word_index.items()}

# load the pre-trained model 
model = load_model('simple_rnn_imdb.h5')

def decode_review(encoded_review):
    return ' '.join([reverse_word_index.get(i - 3, '?') for i in encoded_review])

# function to preprocess user input
def preprocess_text(text):
    words = text.lower().split()
    encoded_review = [word_index.get(word,2) + 3 for word in words]
    padded_review = sequence.pad_sequences([encoded_review], maxlen=500)
    return padded_review

# prediction function
# def predict_sentiment(review):
#    prepprocessed_input = preprocess_text(review)
#    predict = model.predict(prepprocessed_input)
#    sentiment = 'Positive' if predict[0][0] > 0.5 else 'Negative'
#    return sentiment, predict[0][0]

st.title('IMDB Movie Review Sentiment Analysis')
st.write('Enter a movie review to classify it as postive or negative')

user_input = st.text_area('Movie Review')

if st.button('Classify'):
    preprocessed_input = preprocess_text(user_input)
    prediction = model.predict(preprocessed_input)
    sentiment = 'Positive' if prediction[0][0] > 0.5 else 'Negative'

    st.write(f'Sentiment:', {sentiment})
    st.write(f"Prediction score: {prediction[0][0]}")

else:
    st.write("Please enter a movie review and click on the 'Classify' button")