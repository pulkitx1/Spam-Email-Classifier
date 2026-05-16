from flask import Flask, render_template, request
import pickle
import nltk
from nltk.corpus import stopwords
import string

# Initialize Flask app
app = Flask(__name__)

# Load the saved model and vectorizer
# These were created by running train.py
try:
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        tfidf = pickle.load(f)
except FileNotFoundError:
    print("Error: model.pkl or vectorizer.pkl not found. Please run train.py first.")

def preprocess_text(text):
    """
    Same preprocessing function as used in train.py.
    """
    text = text.lower()
    text = "".join([char for char in text if char not in string.punctuation])
    stop_words = set(stopwords.words('english'))
    words = text.split()
    cleaned_words = [word for word in words if word not in stop_words]
    return " ".join(cleaned_words)

@app.route('/')
def home():
    # Render the main page (index.html)
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        # Get message from the form
        message = request.form['message']
        
        # 1. Preprocess the input message
        cleaned_message = preprocess_text(message)
        
        # 2. Vectorize the message using our saved TF-IDF vectorizer
        vectorized_message = tfidf.transform([cleaned_message])
        
        # 3. Predict using our saved model
        prediction = model.predict(vectorized_message)
        
        # 4. Result logic (0 = ham, 1 = spam)
        result = "Spam" if prediction[0] == 1 else "Not Spam"
        
        return render_template('index.html', prediction_text=result, original_message=message)

if __name__ == "__main__":
    # Run the Flask server
    app.run(debug=True)
