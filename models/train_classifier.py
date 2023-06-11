import sys
# import libraries
import re
import pandas as pd
import numpy as np
import sqlalchemy
from sqlalchemy import create_engine
import itertools

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer,TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.base import BaseEstimator, TransformerMixin

import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords

nltk.download('wordnet')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('averaged_perceptron_tagger')

import joblib

import warnings
warnings.filterwarnings('ignore')

class AfterVerbTagExtractor(BaseEstimator, TransformerMixin):
    def __init__(self,tags_list):
        self.tags_list=tags_list
        self.verb_tags=['VB', 'VBP','VBD','VBG','VBN','VBZ']
    def after_verb_tags(self,text):
        sentence_list = nltk.sent_tokenize(text)
        for sentence in sentence_list:
            pos_tags = nltk.pos_tag(tokenize(sentence))
            if len(pos_tags) > 0:
                tags = [tag for _,tag in pos_tags]
                index = [i for i, j in enumerate(tags) if j in self.verb_tags]
                if len(index)> 0:
                    if len(pos_tags) > index[0]+1:
                        tag_after_verb = pos_tags[index[0]+1]
                        if tag_after_verb in self.tags_list:
                            return True
        return False

    def fit(self, x, y=None):
        return self

    def transform(self, X):
        X_tagged = pd.Series(X).apply(self.after_verb_tags)
        return pd.DataFrame(X_tagged)

def load_data(database_filepath):
	# load data from database
	engine = create_engine(f"sqlite:///{database_filepath}")
	df = pd.read_sql_table(con=engine,table_name="disaster_tweet")
	X = df['message']
	Y = df.iloc[:,3:].values
	category_names = df.iloc[:,3:].columns
	return X,Y,category_names


def tokenize(text):
    # Remove non alphanumeric char
    text = re.sub(r'[^a-zA-Z0-9 ]', '', text)
    

    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))

    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
        
    clean_tokens = [w for w in clean_tokens if not w.lower() in stop_words]
    return clean_tokens


def build_model():
	verb_tags = ['VB', 'VBP','VBD','VBG','VBN','VBZ']
	noun_tags = ['NN', 'NNS','NNP','NNPS']
	adjective_tags = ['JJ','JJR','JJS']
	pronoun_tags = ['PRP','PRP']
	adverb_tags = ['RB','RBR','RBS']
	
	pipeline = Pipeline([
		('features', FeatureUnion([
			# TFIDF Features
			('text_pipeline', Pipeline([
				('vect', CountVectorizer(tokenizer=tokenize,ngram_range=(1,2))),
				('tfidf', TfidfTransformer())
			])),
			('first_tag_after_verb', FeatureUnion([
				('verb_after_verb', AfterVerbTagExtractor(tags_list=verb_tags)),
				('noun_after_verb', AfterVerbTagExtractor(tags_list=noun_tags)),
				('adj_after_verb', AfterVerbTagExtractor(tags_list=adjective_tags)),
				('pronoun_after_verb', AfterVerbTagExtractor(tags_list=pronoun_tags)),
				('adverb_after_verb', AfterVerbTagExtractor(tags_list=adverb_tags)),
			])),
		])),
		('classifier', MultiOutputClassifier(RandomForestClassifier(n_estimators=20,min_samples_split=2))),
	])
	return pipeline


def evaluate_model(model, X_test, Y_test, category_names):
	Y_pred = model.predict(X_test)
	for i in range(Y_test.shape[1]):
		print(category_names[i])
		print(classification_report(Y_test.T[i], Y_pred.T[i]))


def save_model(model, model_filepath):
	joblib.dump(model, model_filepath)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()