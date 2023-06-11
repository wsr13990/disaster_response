import json
import plotly
import pandas as pd

import nltk
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords

from flask import Flask
from flask import render_template, request, jsonify
from plotly.graph_objs import Bar
# from sklearn.externals import joblib
import joblib
from sqlalchemy import create_engine

from sklearn.base import BaseEstimator, TransformerMixin
import re


app = Flask(__name__)

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

# load data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('disaster_tweet', engine)

# load model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():
    
    # extract data needed for visuals
    # TODO: Below is an example - modify to extract data for your own visuals
    genre_counts = df.groupby('genre').count()['message']
    genre_names = list(genre_counts.index)

    aid_needed = df.iloc[:,3:].columns

    union_list = []
    for col in aid_needed:
        df_to_append = df[[col]]
        df_to_append.columns = ["value"]
        df_to_append["aid_type"] = col.replace("_", " ")
        union_list.append(df_to_append)
    df_union_all= pd.concat(union_list,ignore_index=True)
    aid_counts = df_union_all.groupby('aid_type').sum()['value']
    aid_needed = [col.replace("_", " ") for col in aid_needed]
    
    # create visuals
    # TODO: Below is an example - modify to create your own visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=genre_names,
                    y=genre_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Message Genres',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=aid_needed,
                    y=aid_counts
                )
            ],

            'layout': {
                'title': 'Distribution of Aid Needed',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Aid Needed"
                }
            }
        }
    ]
    
    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)
    
    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)


# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '') 

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file. 
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3000, debug=True)


if __name__ == '__main__':
    main()