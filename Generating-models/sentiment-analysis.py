import pandas as pd
import numpy as np
import os.path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

ara_corpus_files = [ './corpus_files/sentiment_data_ARA_pos.txt', './corpus_files/sentiment_data_ARA_neg.txt' ]

tun_corpus_files = [ './corpus_files/sentiment_data_TUN_pos.txt', './corpus_files/sentiment_data_TUN_neg.txt' ]


def read_text_file(filename):
    print("Reading file " + filename + "...")
    with open(BASE_DIR + "/" + filename, "r", encoding='utf8') as textfile:
        lines = (line.strip() for line in textfile)
   
    print("File contains f{len(L)} lines.\n")
    yield lines

ara_corpus = [*read_text_file(ara_corpus_files[i]) for i in range(2)]
ara_corpus_sentiment = len(ara_corpus_pos)*[1] + len(ara_corpus_neg)*[-1]

tun_corpus = [*read_text_file(tun_corpus_files[i]) for i in range(2)]
tun_corpus_sentiment = len(tun_corpus_pos)*[1] + len(tun_corpus_neg)*[-1]

"""
Text Preprocessing & Cleaning
"""

import re
import html

# regexp for word elongation: matches 3 or more repetitions of a word character.
two_plus_letters_RE = re.compile(r"(\w)\1{1,}", re.DOTALL)
three_plus_letters_RE = re.compile(r"(\w)\1{2,}", re.DOTALL)
# regexp for repeated words
two_plus_words_RE = re.compile(r"(\w+\s+)\1{1,}", re.DOTALL)


def cleanup_text(text):
    # Remove URLs
    text = re.sub('((www\.[^\s]+)|(https?://[^\s]+))', '', text)

    # Remove user mentions of the form @username
    text = re.sub('@[^\s]+', '', text)
    
    # Replace special html-encoded characters with their ASCII equivalent, for example: &#39 ==> '
    if re.search("&#",text):
        text = html.unescape(text)

    # Remove special useless characters such as _x000D_
    text = re.sub(r'_[xX]000[dD]_', '', text)

    # Replace all non-word characters (such as emoticons, punctuation, end of line characters, etc.) with a space
    text = re.sub('[\W_]', ' ', text)

    # Remove redundant white spaces
    text = text.strip()
    text = re.sub('[\s]+', ' ', text)

    # normalize word elongations (characters repeated more than twice)
    text = two_plus_letters_RE.sub(r"\1\1", text)

    # remove repeated words
    text = two_plus_words_RE.sub(r"\1", text)

    return text

# Apply this function to each document in the corpus
ara_corpus_clean = [cleanup_text(doc) for doc in ara_corpus]

tun_corpus_clean = [cleanup_text(doc) for doc in tun_corpus]


# Quick method: remove documents that contain more than 40% latin characters
MAX_LAT_FRAC = 0.3
ara_corpus_clean = [doc for doc in ara_corpus_clean if (len(re.findall('[a-zA-Z]',doc)) / len(doc)) < MAX_LAT_FRAC]

tun_corpus_clean = [doc for doc in tun_corpus_clean if (len(doc)!=0 and (len(re.findall('[a-zA-Z]',doc)) / len(doc)) < MAX_LAT_FRAC )]

# Letter normalization

def normalizeArabic(corpus):
    corpus = re.sub("ة", "ت", corpus)
    corpus = re.sub("ض", "ظ", corpus)
    corpus = re.sub("ى", "ي", corpus)
    corpus = re.sub("ؤ", "ء", corpus)
    corpus = re.sub("ئ", "ء", corpus)
    corpus = re.sub("[إأٱآا]", "ا", corpus)
    return(corpus)

ara_corpus_clean =  [normalizeArabic(doc) for doc in ara_corpus_clean]

tun_corpus_clean =  [normalizeArabic(doc) for doc in tun_corpus_clean]



## Tokenization  

from nltk.tokenize import RegexpTokenizer

tokenizer = RegexpTokenizer('[^_\W]+')
ara_corpus_tokenized = [tokenizer.tokenize(doc) for doc in ara_corpus_clean]

tun_corpus_tokenized = [tokenizer.tokenize(doc) for doc in tun_corpus_clean]


# Remove stop words -- based on a 'standard' list of stopwords for the Arabic language.

import nltk
nltk.download('stopwords')
# Load stop words from NLTK library
from nltk.corpus import stopwords
stop_words_ar = stopwords.words('arabic')

set(stop_words_ar) & {'من','إلى','عن','على','في','ب','ل','ك','و'}
stop_words_ar = stop_words_ar + ['من','إلى','عن','على','في','ب','ل','ك','و']

# For each document, remove stop words
ara_corpus_tokenized = [[word for word in doc if word not in stop_words_ar] for doc in ara_corpus_tokenized]

tun_corpus_tokenized = [[word for word in doc if word not in stop_words_ar] for doc in tun_corpus_tokenized]

 
# Stemming

import argparse
from nltk.stem.isri import ISRIStemmer

def light_stem(text):
    words = text.split()
    result = list()
    stemmer = ISRIStemmer()
    for word in words:
        word = stemmer.norm(word, num=1)      # remove diacritics which representing Arabic short vowels
        if not word in stemmer.stop_words:    # exclude stop words from being processed
            word = stemmer.pre32(word)        # remove length three and length two prefixes in this order
            word = stemmer.suf32(word)        # remove length three and length two suffixes in this order
            word = stemmer.waw(word)          # remove connective ‘و’ if it precedes a word beginning with ‘و’
            word = stemmer.norm(word, num=2)  # normalize initial hamza to bare alif
        result.append(word)
    return ' '.join(result)

ara_corpus_clean =  [light_stem(doc) for doc in ara_corpus_clean]
ara_corpus_tokenized =  [tokenizer.tokenize(doc) for doc in ara_corpus_clean]


tun_corpus_clean =  [light_stem(doc) for doc in tun_corpus_clean]
tun_corpus_tokenized =  [tokenizer.tokenize(doc) for doc in tun_corpus_clean]


# Remove words that are too short or too long.

ara_distinct_words = {word for doc in ara_corpus_tokenized for word in doc}

ara_corpus_tokenized = [[word for word in doc if len(word)>=4] for doc in ara_corpus_tokenized]

ara_corpus_tokenized = [[word for word in doc if len(word)<=12] for doc in ara_corpus_tokenized]


tun_distinct_words = {word for doc in tun_corpus_tokenized for word in doc}

tun_corpus_tokenized = [[word for word in doc if len(word)>=4] for doc in tun_corpus_tokenized]

tun_corpus_tokenized = [[word for word in doc if len(word)<=12] for doc in tun_corpus_tokenized]


"""
Document Representation
"""

# First, concatenate the words in the cleaned corpus (because BOW method in scikit-learn requires this format)
ara_corpus_bow = [' '.join(doc) for doc in ara_corpus_tokenized]

tun_corpus_bow = [' '.join(doc) for doc in tun_corpus_tokenized]


# Build the vocabulary set
from sklearn.feature_extraction.text import TfidfVectorizer

max_words = 10000
maxdf = 0.7
mindf = 5

# create an instance of this class
bow_model = TfidfVectorizer(max_df=maxdf, min_df=mindf, max_features=max_words, stop_words=[], use_idf = True)

bow_model.fit( ara_corpus_bow )

ara_bow_dtm = bow_model.transform(ara_corpus_bow)


tun_bow_model = TfidfVectorizer(max_df=maxdf, min_df=mindf, max_features=max_words, stop_words=[], use_idf = True)

tun_bow_model.fit( tun_corpus_bow )

tun_bow_dtm = tun_bow_model.transform(tun_corpus_bow)


from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn import metrics

## Remove documents that do not contain any vocabulary terms

nb_terms_per_doc = np.array((ara_bow_dtm>0).sum(axis=1))  # calculate sum of rows of DTM matrix
nb_terms_per_doc = nb_terms_per_doc.ravel() 

idx = nb_terms_per_doc>0
ara_bow_dtm_filt = ara_bow_dtm[nb_terms_per_doc>0,:]
ara_corpus_bow_filt = [ara_corpus_bow[i] for i,x in enumerate(idx) if x]
ara_corpus_sentiment_filt = [ara_corpus_sentiment[i] for i,x in enumerate(idx) if x]

X = ara_bow_dtm_filt
y = ara_corpus_sentiment_filt


# TUN #

tun_nb_terms_per_doc = np.array((tun_bow_dtm>0).sum(axis=1))  # calculate sum of rows of DTM matrix
tun_nb_terms_per_doc = tun_nb_terms_per_doc.ravel() 

tun_idx = tun_nb_terms_per_doc>0
tun_bow_dtm_filt = tun_bow_dtm[tun_nb_terms_per_doc>0,:]
tun_corpus_bow_filt = [tun_corpus_bow[i] for i,x in enumerate(tun_idx) if x]
tun_corpus_sentiment_filt = [tun_corpus_sentiment[i] for i,x in enumerate(tun_idx) if x]

tun_X = tun_bow_dtm_filt
tun_y = tun_corpus_sentiment_filt


# Split the data into training and testing

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state=1996)


tun_X_train, tun_X_test, tun_y_train, tun_y_test = train_test_split(tun_X, tun_y, test_size = 0.3, random_state=1996)



"""
Train classifier using Naive Bayes
"""

NB_model = MultinomialNB(alpha = 1.0)
NB_model.fit(X_train, y_train)

y_pred_NB = NB_model.predict(X_test)


tun_NB_model = MultinomialNB(alpha = 1.0)
tun_NB_model.fit(tun_X_train, tun_y_train)

tun_y_pred_NB = tun_NB_model.predict(tun_X_test)

#print(metrics.accuracy_score(y_test, y_pred_NB))

#print(metrics.confusion_matrix(y_test, y_pred_NB))


"""
RETRAIN ARA
"""
ara_NB_model = MultinomialNB(alpha = 1.0)
ara_NB_model.fit(X, y)


"""
RETRAIN TUN
"""
tun_NB_model = MultinomialNB(alpha = 1.0)
tun_NB_model.fit(tun_X, tun_y)



"""
Dumping our models
"""

import pickle
import os

#pkl_objects - where our classifier and other objects will be stored
dest = os.path.join(BASE_DIR,'../Web-application/pkl_objects/sentiment')

if not os.path.exists(dest):
    os.makedirs(dest)

pickle.dump(ara_NB_model,open(os.path.join(dest,'aranbmodel.pkl'),'wb'),protocol=4)
pickle.dump(tun_NB_model,open(os.path.join(dest,'tunnbmodel.pkl'),'wb'),protocol=4)

pickle.dump(bow_model,open(os.path.join(dest,'arabowmodel.pkl'),'wb'),protocol=4)
pickle.dump(tun_bow_model,open(os.path.join(dest,'tunbowmodel.pkl'),'wb'),protocol=4)




"""
# To test an input locally, uncomment these lines: 

# Modify the user_input and lang with the data you want to test
user_input=pd.Series(['لمحادثات المخفية المحادثات المخفية الخاصية دي اتلغت ليه ممكن حد يفهمني'])
lang="ARA"


if lang == 'ARA':
	dtm=bow_model.transform(user_input)
	print(ara_NB_model.predict(dtm))
elif lang == 'TUN':
	dtm=tun_bow_model.transform(user_input)
	print(tun_NB_model.predict(dtm))

"""

