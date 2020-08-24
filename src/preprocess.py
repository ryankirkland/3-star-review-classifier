import re
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

wordnet = WordNetLemmatizer()
nltk.download('stopwords')

def remove_punc(string:str) -> str:
    '''Given a string, removes all punctuation and returned punctuation-less string'''
    return re.sub(r"(@[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", string)

def tokenize(str):
    '''
    Tokenize a str and return a tokenized list.
    '''
    return [word for word in word_tokenize(str)]

def lemmatize(doc):
    '''Takes in a doc and lemmatizes tokens in doc
    Parameters
    ----------
    doc: list of tokens
    
    Returns
    -------
    lemmatized tokens
    '''
    return [wordnet.lemmatize(tkn) for tkn in doc]

def rm_stop_words(doc, stops=set(stopwords.words('english'))):
    '''Takes in a doc and removes stop words
    Parameters
    ----------
    doc: list of tokens
    
    Returns
    -------
    Tokens with stop words removed
    '''
    return([w for w in doc if w not in stops])

def preprocess_corpus(content):
    '''
    Add docstring. Make flexible to allow for doing, or not doing, preprocessing functions. 
    Parameters
    ----------
    content (str): a collection of strings
    Returns
    -------
    A list of lists: each list contains a tokenized version of the original string
    '''
    preprocessed = []
    for i in range(len(content)):
        step_1 = remove_punc(content[i].lower())
        step_2 = tokenize(step_1)
        step_3 = lemmatize(step_2)
        step_4 = rm_stop_words(step_3)
        preprocessed.append(step_4)
    return preprocessed