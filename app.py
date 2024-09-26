import streamlit as st
import pandas as pd
import pickle  # or import your specific model library, e.g. tensorflow, torch
from sklearn.feature_extraction.text import CountVectorizer  # Example if you're working with text data

# Load your pre-trained model (replace this with your actual model loading code)
def load_model():
    with open('LR_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

# Function to preprocess the input (modify according to your model's needs)
def preprocess_input(input_text, vectorizer):
    # Assuming your model expects a vectorized text input
    return vectorizer.transform([input_text])

# Load vectorizer (if applicable for text data)
def load_vectorizer():
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return vectorizer

# Set up the Streamlit app
st.title("Hate Speech Detection Model")

# Load the pre-trained model and vectorizer
model = load_model()
vectorizer = load_vectorizer()

# Create a text input field for user to enter the text
user_input = st.text_area("Enter a tweet to analyze", "")

if st.button('Predict'):
    if user_input:
        # Preprocess the input text
        processed_input = preprocess_input(user_input, vectorizer)
        
        # Make a prediction using the loaded model
        prediction = model.predict(processed_input)
        
        # Map the prediction to a human-readable label
        label_map = {0: 'Hate Speech', 1: 'Offensive Language', 2: 'Neither'}
        result = label_map[prediction[0]]
        
        # Display the prediction
        st.subheader(f"Prediction: {result}")
    else:
        st.error("Please enter text for prediction!")

# Optional: Upload CSV for batch predictions
uploaded_file = st.file_uploader("Choose a CSV file for batch prediction")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    if 'text_column' in df.columns:  # Replace 'text_column' with the actual column name in your CSV
        # Preprocess and predict for each text in the file
        processed_data = vectorizer.transform(df['text_column'])
        predictions = model.predict(processed_data)
        
        # Add predictions to the DataFrame
        df['prediction'] = [label_map[pred] for pred in predictions]
        
        # Display results
        st.write(df[['text_column', 'prediction']])
    else:
        st.error("The CSV must have a 'text_column' column!")
