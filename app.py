import streamlit as st
import pickle

# Load the pre-saved model and vectorizer
with open('model_with_vectorizer.pkl', 'rb') as file:
    model, vectorizer = pickle.load(file)

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
        st.write(input_vector)
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
