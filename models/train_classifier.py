import sys
import joblib
import re
import pandas as pd
from sqlalchemy import create_engine
import sqlite3

import nltk
nltk.download('stopwords')
nltk.download('punkt')
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.snowball import SnowballStemmer

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


# List of stopwords
stop = stopwords.words('english')


def load_data(database_filepath):

    '''
    Load in the clean dataset from the SQLite database.

    Args:
        database_filepath (str): path to the SQLite database

    Returns:
        (Tuple(Pandas series, Pandas df, list))
            Messages data
            Target labels
            Names of target labels
    '''

    conn = sqlite3.connect('D:\Protege_Code\disaster-response\data\messages.db')

    # c = conn.cursor()
    # engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql('SELECT * FROM messages', conn)
    print(df)
    X = df['message'].copy()
    Y = df.iloc[:, 4:].copy()
    categories = Y.columns.tolist()
    return X, Y, categories

    
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


def build_model():

    '''
    Build a machine learning pipeline that converts text data into a numeric vector then classifies multiple binary
    target labels.

    Steps:
        1. TfidfVectorizer: vectorize text data using term frequency and inverse document frequency
        2. MultiOutputClassifier(LogisticRegression): classify multiple labels using logistic regression

    Args:
        None

    Returns:
        (Sklearn pipeline) pipeline estimator
    '''

    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 1), max_df=0.2, max_features=1000)),
        ('clf', MultiOutputClassifier(LogisticRegression(max_iter=1000, random_state=0)))
    ])
    return pipeline


def evaluate_model(model, X_test, Y_test, category_names):

    '''
    Evaluate the machine learning model using a test dataset and print the classification report metrics for each label.

    Args:
        model (Sklearn estimator): machine learning model
        X_test (list-like object): test set text data
        Y_test (Pandas dataframe): test set target labels
        category_names (list): names of target labels

    Returns:
        (Pandas dataframe) Classification report metrics
    '''

    pred = pd.DataFrame(model.predict(X_test), columns=category_names)

    metrics = []

    for col in category_names:
        report = classification_report(Y_test[col], pred[col])
        scores = report.split('accuracy')[1].split()
        metrics.append([float(scores[i]) for i in [0, 4, 5, 6, 10, 11, 12]])

    metric_names = ['accuracy', 'macro_avg_precision', 'macro_avg_recall', 'macro_avg_f1', 'weighted_avg_precision',
                    'weighted_avg_recall', 'weighted_avg_f1']
    metrics_df = pd.DataFrame(metrics, columns=metric_names, index=category_names)

    print(metrics_df)
    print(metrics_df.sum)
    return metrics_df
        

def save_model(model, model_filepath):

    '''
    Save the machine learning model as a pickle file.

    Args:
        model (Sklearn estimator): machine learning model
        model_filepath (str): path to save the model

    Returns:
        None
    '''

    joblib.dump(model, model_filepath)
    return


def main():

    '''
    This file is the ML pipeline that trains the classifier and saves it as a pickle file.

    From this project's root directory, run this file with:
    python models/train_classifier.py data/messages.db models/classifier.pkl
    '''

    database_filepath = 'data/messages.db'
    model_filepath = 'classifier.pkl'

    print('Loading data...\n    DATABASE: {}'.format(database_filepath))
    X, Y, category_names = load_data(database_filepath)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.25, random_state=0)

    print('Building model...')
    model = build_model()

    print('Training model...')
    model.fit(X_train, Y_train)

    print('Evaluating model...')
    evaluate_model(model, X_test, Y_test, category_names)

    print('Saving model...\n    MODEL: {}'.format(model_filepath))
    save_model(model, model_filepath)

    print('Trained model saved!')


if __name__ == '__main__':
    main()