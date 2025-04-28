# 🔮 Next Word Prediction using LSTM

This project builds a **Next Word Prediction** app using an **LSTM (Long Short-Term Memory)** model trained on Shakespeare's *Hamlet*. Given a sentence and the number of next words required, the model predicts and completes the text. The app is deployed with **Streamlit** for an interactive user experience.

## 🧠 Model Summary

- Frameworks: TensorFlow, Keras
- Architecture:
  - Embedding Layer
  - LSTM Layer
  - Dropout Layer
  - Dense Layer (Softmax activation)
- Training Data: Hamlet by William Shakespeare
- Input Sequence Length: 14 words
- Callbacks: EarlyStopping, ReduceLROnPlateau, TensorBoard
- Model Output: Next word predictions based on input text

## ✅ Features

- Streamlit-based interactive UI for user input and prediction
- Tokenizer saved using pickle for efficient text preprocessing
- TensorBoard integration for training visualization
- Dynamic learning rate reduction and early stopping to optimize training
- Support for predicting multiple next words in one go

## 📦 Requirements

Install dependencies with:
requirements.txt (pip install -r requirements.txt)

## 🚀 How to Run

1. Train the model (if not already trained):
   - Open and run all cells in:

## Streamlit App
streamlit run app.py
- Enter a sentence.
- Specify how many next words you want.
- Click **Submit** to get the completed text.

## 📊 Output

- Streamlit app predicts and displays the completed sentences interactively

## 📁 Project Files

- `app.py` — Streamlit app for user interaction
- `experiment_v1.ipynb` — Jupyter Notebook for model training
- `next_word_pred_lstmmodel.h5` — Trained LSTM model
- `tokenizer.pkl` — Tokenizer object for input text processing
- `logs/` — TensorBoard training logs
- `hamlet.txt` — Dataset used for training

