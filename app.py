
# Import necessary libraries
import string
import streamlit as st
import pickle
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer

# Download NLTK data for stopwords and tokenizer model
nltk.download('stopwords')
nltk.download('punkt')

# Define a function to preprocess text
def transform_text(text):
    # Make text lowercase
    text = text.lower()

    # Tokenize the text
    text = nltk.word_tokenize(text)

    # Remove special characters
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    # Remove stopwords and punctuation
    text = y[:]
    y.clear()
    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    # Stem the words
    text = y[:]
    y.clear()
    ps = PorterStemmer()
    for i in text:
        y.append(ps.stem(i))

    # Return the preprocessed text as a string
    return " ".join(y)


# Load the TF-IDF vectorizer with error handling
try:
    tfidf = pickle.load(open('F:/Machine Learning Projects/sms classification/vectorizer.pkl', 'rb'))
    model = pickle.load(open('F:/Machine Learning Projects/sms classification/model.pkl', 'rb'))
except FileNotFoundError:
    st.error("Error: vectorizer.pkl file not found.")
except Exception as e:
    st.error(f"An error occurred while loading vectorizer.pkl: {str(e)}")



# Streamlit app title
st.title("Email/ SMS Spam Classifier")

# Text input for user to enter a message
input_sms = st.text_input("Enter the message")

# Button to trigger prediction
if st.button("Predict"):
    # Preprocess the input message
    transformed_sms = transform_text(input_sms)

    # Vectorize the preprocessed message using TF-IDF
    vector_input = tfidf.transform([transformed_sms])

    # Make a prediction using the loaded model
    result = model.predict(vector_input)[0]

    # Display the prediction result
    if result == 1:
        st.header('Spam')
    else:
        st.header("Not Spam")
