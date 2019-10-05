import pandas as pd
import numpy as np

from matplotlib import pyplot as plt
import os.path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


# set the font size of plots
plt.rcParams['font.size'] = 14
np.set_printoptions(precision=3)

"""
Read Raw Data
"""
corpus_files = ['./corpus_files/langid_data_TUN-AR.txt', './corpus_files/langid_data_ARA.txt', 'corpus_files/sentiment_data_TUN_neg.txt', 'corpus_files/sentiment_data_TUN_pos.txt', 'corpus_files/sentiment_data_ARA_neg.txt', 'corpus_files/sentiment_data_ARA_pos.txt']

def read_text_file(filename):
    print('Reading file ' + filename + "...")
    with open(BASE_DIR + "/" + filename, "r", encoding='utf8') as textfile:   #, encoding='utf8'
        L = []
        for line in textfile:
            L.append(line.strip())
        print('File contains ', len(L), "lines.\n")
        return L

tun_corpus = read_text_file(corpus_files[0])
tun_corpus1 = read_text_file(corpus_files[2])
tun_corpus2 = read_text_file(corpus_files[3])

ara_corpus = read_text_file(corpus_files[1])
ara_corpus1 = read_text_file(corpus_files[4])
ara_corpus2 = read_text_file(corpus_files[5])


"""
Text Cleaning function
"""
import re

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
    #if re.search("&#",text):
        #text = html.unescape(text)

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

# unit test of this function


"""
TEXT CLEANING
"""
tun_corpus_clean = [cleanup_text(doc) for doc in tun_corpus]
tun_corpus_clean1 = [cleanup_text(doc) for doc in tun_corpus1]
tun_corpus_clean2 = [cleanup_text(doc) for doc in tun_corpus2]
tun_corpus_clean = tun_corpus_clean + tun_corpus_clean1 + tun_corpus_clean2

ara_corpus_clean = [cleanup_text(doc) for doc in ara_corpus]
ara_corpus_clean1 = [cleanup_text(doc) for doc in ara_corpus1]
ara_corpus_clean2 = [cleanup_text(doc) for doc in ara_corpus2]
ara_corpus_clean = ara_corpus_clean + ara_corpus_clean1 + ara_corpus_clean2


"""
Remove all documents that contain a large fraction of latin characters (for example more than 70%)
"""
tun_corpus_clean_2 = []
for line in tun_corpus_clean:
	if(len(line)!=0 and len(re.findall('[a-zA-Z]',line))/len(line) < 0.3 ):
		tun_corpus_clean_2.append(line)
ara_corpus_clean_2 = []
for line in ara_corpus_clean:
	if(len(line)!=0 and len(re.findall('[a-zA-Z]',line))/len(line) < 0.3 ):
		ara_corpus_clean_2.append(line) 

#print(len(tun_corpus_clean_2))


"""
remove very short docs
"""
tun_corpus_clean_3 = []
for line in tun_corpus_clean_2:
    if(len(line))>10 :
        tun_corpus_clean_3.append(line)
ara_corpus_clean_3= []
for line in ara_corpus_clean_2:
   if(len(line))>10 :
        ara_corpus_clean_3.append(line)



"""
devide each corpus into train_corpus and test_corpus
"""

from sklearn.model_selection import train_test_split
#?train_test_split

tun_corpus_clean_train, tun_corpus_clean_test = train_test_split(tun_corpus_clean,test_size=0.3 )

ara_corpus_clean_train, ara_corpus_clean_test = train_test_split(ara_corpus_clean,test_size=0.3 )



# create data frame
train_df = pd.DataFrame({'document':[], 'language':[]})

# fill the language column
train_df.language = pd.Series(['TUN']*len(tun_corpus_clean_train) + ['ARA']*len(ara_corpus_clean_train))


# fill the document column -- CONCATENATE the TUN CORPUS and ARA CORPUS
train_df.document = tun_corpus_clean_train+ara_corpus_clean_train

# create data frame
test_df = pd.DataFrame({'document':[], 'language':[]})

# fill the language column
test_df.language = pd.Series(['TUN']*len(tun_corpus_clean_test) + ['ARA']*len(ara_corpus_clean_test))


# fill the document column -- CONCATENATE the TUN CORPUS and ARA CORPUS
test_df.document = tun_corpus_clean_test+ara_corpus_clean_test
print(len(tun_corpus_clean_test), len(ara_corpus_clean_test))


"""
convertion into numetic feature vectors using the BOW tfidf method with character ngrams
"""
from sklearn.feature_extraction.text import TfidfVectorizer

n = 4 # hyperparameter for of character ngrams ; you can change it if you want but n=3 is a reasonable value ...

# Create an instance of TfidfVectorizer class with analyzer = 'char' so that it generates bag of characters and not bag of words

bow_model_char = TfidfVectorizer(analyzer='char', ngram_range=(1,n), max_df = 0.85, min_df= 0.25)

print("n =", str(n)," ,max_df =", str(bow_model_char.max_df), " ,min_df=", str(bow_model_char.min_df))
# Call fit method with the combined training corpus
bow_model_char.fit( train_df.document )

# Create DTM matrix of the combined training corpus and test corpus
train_dtm= bow_model_char.transform(train_df.document)
test_dtm = bow_model_char.transform(test_df.document)


"""
Apply MultinomialNB
"""

from sklearn.naive_bayes import MultinomialNB

nb_model = MultinomialNB()

nb_model.fit(train_dtm,train_df.language)

y_pred_class = nb_model.predict(test_dtm)


"""
evaluating performance or the classifier: accuracy and confusion matrix
"""
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

print(accuracy_score(test_df.language, y_pred_class))

print(confusion_matrix(test_df.language, y_pred_class))


"""
Dumping our models
"""

import pickle
import os

#pkl_objects - where our classifier and other objects will be stored
dest = os.path.join(BASE_DIR,'../Web-application/pkl_objects/language')

if not os.path.exists(dest):
    os.makedirs(dest)

pickle.dump(nb_model,open(os.path.join(dest,'nbmodel.pkl'),'wb'),protocol=4)

pickle.dump(bow_model_char,open(os.path.join(dest,'bowmodelchar.pkl'),'wb'),protocol=4)



"""
# To test an input locally, uncomment these lines: 

import langdetect
from langdetect.lang_detect_exception import LangDetectException

input_= "الشمال الغربي ينتخبون قلب تونس..."

try:
    res = langdetect.detect_langs(input_)   # LANGDETECT
    #res = langdetect.detect(test_doc) 
except LangDetectException:
    res = langdetect.language.Language("UNK",0)
result=str(res[0]).split(':')[0]

if result != "ar":
	print("OTH")
else:
	input_dtm = bow_model_char.transform( [input_])
	print(nb_model.predict(input_dtm))
"""

