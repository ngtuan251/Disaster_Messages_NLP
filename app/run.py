import joblib
import json
import re
import pandas as pd
from sqlalchemy import create_engine

import plotly
from plotly.graph_objs import Bar, Histogram

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer
from sklearn.feature_extraction.text import TfidfVectorizer

from flask import Flask
from flask import request, jsonify, render_template


# Create app
app = Flask(__name__)


# List of stopwords
stop = stopwords.words('english')


def get_top_words(txt, num_words=10):

    '''
    Find words with the highest document frequency.

    Args:
        txt (list-like object): text data
        num_words (int): number of words to find

    Returns:
        (Pandas series) top words and their document frequencies
    '''

    tfidf = TfidfVectorizer(stop_words=stop, max_features=num_words)
    tfidf.fit(txt)
    words = tfidf.vocabulary_
    
    for word in words:
        words[word] = txt[txt.str.contains(word)].count()
    return pd.Series(words).sort_values()


def tokenize(text):

    '''
    Tokenize a string into word stems and remove stopwords.

    Steps:
        1. Lowercase characters
        2. Remove punctuation
        3. Tokenize
        4. Strip white spaces
        5. Remove stopwords
        6. Stem words

    Args:
        text (str): Text to tokenize

    Returns:
        (list) stemmed non-stopword tokens
    '''
    
    # Steps 1 - 3
    tokens = word_tokenize(re.sub(r'[^A-Za-z0-9]', ' ', text.lower()))
    
    # Step 4 - 5
    stopwords_removed = [word.strip() for word in tokens if word.strip() not in stop]
    
    # Step 6
    stemmer = SnowballStemmer('english')
    return [stemmer.stem(word) for word in stopwords_removed]


# Load data
engine = create_engine('sqlite:///../data/messages.db')
df = pd.read_sql_table('messages', engine)


# Load model
model = joblib.load("../models/classifier.pkl")


# Home page
@app.route('/')
@app.route('/home')
@app.route('/index')
def index():
    
    # Genre counts
    genre_counts = df.groupby('genre').count()['message']
    genre_names = genre_counts.index.tolist()
    
    # Message word counts
    word_counts = df.message.apply(lambda s: len(s.split()))
    word_counts = word_counts[word_counts <= 100]
    
    # Category counts
    cat_counts = df.iloc[:, 4:].sum().sort_values()[-10:]
    cat_names = cat_counts.index.tolist()
    
    # Top words
    top_counts = get_top_words(df.message)
    top_words = top_counts.index.tolist()
    
    # Create visuals
    graphs = [
        
        # Genre counts
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {'title': "Number of messages"},
                'xaxis': {'title': "Genre"}
            }
        },
        
        # Message word counts
        {
            'data': [
                Histogram(
                    x=word_counts
                )
            ],
            'layout': {
                'title': 'Distribution of Message Word Counts',
                'yaxis': {'title': "Number of messages"},
                'xaxis': {'title': "Word count"}
            }
        },
        
        # Category counts
        {
            'data': [
                Bar(
                    x=cat_counts,
                    y=cat_names,
                    orientation='h'
                )
            ],
            'layout': {
                'title': 'Top Message Categories',
                'yaxis': {'title': "Category"},
                'xaxis': {'title': "Number of Messages"},
                'margin': {'l': 100}
            }
        },
        
        # Top words
        {
            'data': [
                Bar(
                    x=top_counts,
                    y=top_words,
                    orientation='h'
                )
            ],
            'layout': {
                'title': 'Most Common Words in Messages',
                'yaxis': {'title': "Word"},
                'xaxis': {'title': "Number of Messages"},
                'margin': {'l': 100}
            }
        }
    ]
    
    # Encode visuals in JSON
    ids = [f'graph-{i}' for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # Render home page
    return render_template('index.html', ids=ids, graphJSON=graphJSON)


# Classification results page
@app.route('/result')
def result():
    
    # Save user input
    query = request.args.get('query', '')

    # Classify message
    classification_labels = model.predict([query])[0]
    classification_result = dict(zip(df.columns[4:], classification_labels))

    # Render result page
    return render_template('result.html', query=query, classification_result=classification_result)


def main():
    app.run(host='0.0.0.0', port=3001, debug=False)


if __name__ == '__main__':
    main()