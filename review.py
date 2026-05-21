# Import libraries
import numpy as np
import tensorflow as tf
import streamlit as st
from tensorflow.keras.datasets import imdb
from tensorflow.keras.preprocessing import sequence
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences

# load the imdb dataset word index
word_index = imdb.get_word_index()
reverse_word_index = {value : key for key,value in word_index.items()}

# load pre trained model with relu activation
model = load_model('simple_rnn_imdb.keras', compile=False)

# Step-2 Helper function
# Function to decode reviews
def decoded_review(encoded_review):
  return ' '.join([reverse_word_index.get(i-3,'?') for i in encoded_review])

# Function for preprocess user input
def preprocess_user_input(user_input):
  words= user_input.lower().split()
  encoded_review= [word_index.get(word,2)+3 for word in words]
  padded_review= sequence.pad_sequences([encoded_review], maxlen=500)
  return padded_review


# Streamlit app
st.title('IMDB Movie Review Sentiment Analyzer')
st.write('Enter a movie review to classify it as Positive or Negative')

# User Input

user_input = st.text_area('Movie Review')

if st.button('Classify'):
  preprocessed_input= preprocess_user_input(user_input)

  # Make prediction
  prediction= model.predict(preprocessed_input)
  sentiment= 'Positive' if prediction[0][0]>0.5 else 'Negative'

  # Display the result
  st.write(f'Sentiment: {sentiment}')
  st.write(f'Prediction Score: {prediction[0][0]}')
else:
  st.write('Please enter a movie review')
  







 


