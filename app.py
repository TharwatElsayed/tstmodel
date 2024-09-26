import streamlit as st
import pickle
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer

# Load the trained Logistic Regression model
model_path = 'LR_model.pkl'
with open(model_path, 'rb') as file:
    model = pickle.load(file)

# Load the dataset to get vocabulary for vectorizer
csv_file_path = 'labeled_data.csv'
data_df = pd.read_csv(csv_file_path)

# Set up CountVectorizer based on the dataset (fit only on 'tweet' column)
vectorizer = CountVectorizer(stop_words='english')
vectorizer.fit(data_df['tweet'])

# Streamlit app setup
st.title("Hate Speech Detection App")
st.write("This app uses a logistic regression model to classify text into one of three categories: "
         "'Hate Speech', 'Offensive Language', or 'Neither'.")

# Text input from user
user_input = st.text_area("Enter the text for classification:")

# When user submits the text
if st.button("Classify"):
    if user_input:
        # Transform the user input using the vectorizer
        input_vector = vectorizer.transform([user_input])
        
        # Predict the class using the logistic regression model
        prediction = model.predict(input_vector)[0]

        # Display the prediction result
        if prediction == 0:
            st.success("The text is classified as: Hate Speech")
        elif prediction == 1:
            st.warning("The text is classified as: Offensive Language")
        else:
            st.info("The text is classified as: Neither")
    else:
        st.error("Please enter some text to classify.")
