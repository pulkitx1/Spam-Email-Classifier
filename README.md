# Spam Email Classifier 📧

A beginner-friendly Machine Learning project created for a first-year engineering student's portfolio. This project uses the **SMS Spam Collection** dataset from Kaggle to classify messages as "Spam" or "Not Spam" (Ham).

## 🚀 Features
- **Algorithm:** Multinomial Naive Bayes
- **Vectorization:** TF-IDF (Term Frequency-Inverse Document Frequency)
- **Frontend:** Simple Flask-based Web Interface
- **Preprocessing:** Lowercase conversion, punctuation removal, and stopword removal using NLTK.

## 🛠️ Tech Stack
- **Language:** Python
- **Libraries:** Pandas, NumPy, Scikit-learn, NLTK
- **Web Framework:** Flask
- **Data Persistence:** Pickle

## 📂 Project Structure
```text
Spam Email Classifier/
├── app.py              # Flask backend server
├── train.py            # Model training script
├── spam.csv            # The dataset
├── model.pkl           # Saved trained model
├── vectorizer.pkl      # Saved TF-IDF vectorizer
├── requirements.txt    # List of dependencies
├── templates/
│   └── index.html      # Frontend HTML
└── static/
    └── style.css       # Frontend Styling
```

## ⚙️ Setup Instructions

### 1. Install Dependencies
Make sure you have Python installed, then run:
```bash
pip install -r requirements.txt
```

### 2. Train the Model
Run the training script to process the data and generate the `.pkl` files:
```bash
python3 train.py
```

### 3. Run the Application
Start the Flask development server:
```bash
python3 app.py
```

### 4. Open in Browser
Visit `http://127.0.0.1:5000` to use the classifier!

## 📊 Dataset
The dataset used is the [SMS Spam Collection](https://www.kaggle.com/datasets/uciml/sms-spam-collection-dataset) from Kaggle. It contains over 5,000 SMS messages labeled as ham (legitimate) or spam.

## 📝 Author
Created as a mini-project for learning the basics of Natural Language Processing (NLP) and Machine Learning.
