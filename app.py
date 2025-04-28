import streamlit as st
import numpy as np
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences

model = tf.keras.models.load_model('next_word_pred_lstmmodel.h5')

with open('tokenizer.pkl','rb') as file:
    tokenizer = pickle.load(file)


# Streamlit App
st.title('Next Word Prediction')
# Input
input = st.text_input('Enter the sentence:', placeholder='Type a sentence here')
number_of_words = st.number_input('Enter the number of next words required:',min_value=1)
# Submit Button
submit_button = st.button("Submit")
    
def prediction(text:str, n_word:int)->str:
    """
    Predict the next 'n_word' words based on the input 'text'.
    
    Args:
    text (str): The input text to seed the prediction.
    n_word (int): The number of words to predict and append.
    
    Returns:
    str: The input text appended with the predicted words.
    """
    
    for _ in range(n_word):
        # Convert the input text to sequences (tokenized)
        token_text = tokenizer.texts_to_sequences([text])[0]
        
        # Pad the sequence to the max sequence length expected by the model
        padded_token_input = pad_sequences([token_text], maxlen=14, padding="pre")
        
        # Predict the probabilities for the next word
        output_prob = model.predict(padded_token_input, verbose=1)
        
        # Find the word with the highest probability
        pos = np.argmax(output_prob)
        
        # Map the predicted index back to the corresponding word
        for word, index in tokenizer.word_index.items():
            if index == pos:
                # Append the predicted word to the input text
                text = text + " " + word
                break
    
    return text

if submit_button:
    text = prediction(input,number_of_words)
    st.write(f"{text}")