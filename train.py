import pandas as pd
import numpy as np
import nltk
from nltk.corpus import stopwords
import string
import pickle
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score

# Step 1: Download necessary NLTK data
# These are needed for text preprocessing like removing common words (stopwords)
print("Downloading NLTK data...")
nltk.download('stopwords')
nltk.download('punkt')

def preprocess_text(text):
    """
    Simple function to clean the text data.
    1. Converts to lowercase
    2. Removes punctuation
    3. Removes stopwords (common words like 'the', 'is', etc.)
    """
    # Convert to lowercase
    text = text.lower()
    
    # Remove punctuation
    # string.punctuation contains: !"#$%&'()*+,-./:;<=>?@[\]^_`{|}~
    text = "".join([char for char in text if char not in string.punctuation])
    
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    
    return " ".join(cleaned_words)

# Step 2: Load the dataset
# Using 'latin-1' encoding because the CSV file might have special characters
print("Loading dataset...")
df = pd.read_csv('spam.csv', encoding='latin-1')

# The Kaggle dataset has extra columns we don't need
# We only need 'v1' (label) and 'v2' (the message)
df = df[['v1', 'v2']]
df.columns = ['label', 'message']

# Step 3: Preprocess the data
print("Preprocessing messages (this might take a moment)...")
df['message'] = df['message'].apply(preprocess_text)

# Convert ham/spam labels to 0 and 1
# ham = 0 (Not Spam), spam = 1 (Spam)
df['label'] = df['label'].map({'ham': 0, 'spam': 1})

# Step 4: Feature Extraction (TF-IDF)
# TF-IDF converts text into numbers that the machine learning model can understand
print("Vectorizing text data...")
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['message'])
y = df['label']

# Step 5: Split the data into Training and Testing sets
# 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 6: Train the model
# Using Multinomial Naive Bayes - a very common algorithm for text classification
print("Training the Multinomial Naive Bayes model...")
model = MultinomialNB()
model.fit(X_train, y_train)

# Step 7: Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"\nTraining Complete!")
print(f"Model Accuracy: {accuracy * 100:.2f}%")

# Step 8: Save the model and vectorizer
# We save these as .pkl files so our Flask app can use them later
print("Saving model and vectorizer...")
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)
with open('vectorizer.pkl', 'wb') as f:
    pickle.dump(tfidf, f)

print("Files saved: model.pkl and vectorizer.pkl")
