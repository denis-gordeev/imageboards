# -*- coding: UTF-8 -*-
import re
import cPickle
import os
import csv
import gensim
import pymorphy2
import pandas as pd
import nltk.data
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.snowball import SnowballStemmer
from gensim.models import word2vec
# Import various modules for string cleaning
# from bs4 import BeautifulSoup

# already_parsed
# specom_b = pd.read_csv( 'specom-b.csv', header=0, delimiter="\t", quoting = csv.QUOTE_MINIMAL )
# specom_p = pd.read_csv( "specom-po.csv", header=0, delimiter="\t", quoting = csv.QUOTE_MINIMAL )
lemmas_bool = ''

boards = ['b', 'po']

def message_to_wordlist(message, lemmas_bool, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    # review_text = BeautifulSoup(review).get_text()
    #
    # 2. Remove messages numbers
    message_text = re.sub(">>\d+", "", message)
    message_text = message_text.lower()
    message_text = re.sub(u"ё", 'e', message_text, re.UNICODE)
    tokenizer = WordPunctTokenizer()
    # 3. Convert words to lower case and split them
    words = tokenizer.tokenize(message_text)
    lemmas = []
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("russian"))
        words = [w for w in words if w not in stops and w.isalpha()]
    if lemmas_bool == 'l':
        for word in words:
            word_parsed = morph.parse(word)
            if len(word_parsed) > 0:
                lemmas.append(word_parsed[0].normal_form)
    elif lemmas_bool == 's':
        for word in words:
            word = stemmer.stem(word)
            if word.isalpha():
                lemmas.append(word)
    else:
        lemmas = words
    # 5. Return a list of words
    lemmas = [l for l in lemmas if l]
    return(lemmas)
    # return(words)


# Define a function to split a message into parsed sentences
def message_to_sentences(message, tokenizer, lemmas_bool, remove_stopwords=True):
    sents = []
    # Function to split a message into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    if type(message) == str and len(message) > 2:
        try:
            message = message.decode('utf-8')
            raw_sentences = tokenizer.tokenize(message.strip())
            # 2. Loop over each sentence
            for raw_sentence in raw_sentences:
                # If a sentence is empty, skip it
                if len(raw_sentence) > 0:
                    # Otherwise, call message_to_wordlist to get a list of words
                    raw_sentence = message_to_wordlist(raw_sentence, lemmas_bool, remove_stopwords)
                    if raw_sentence:
                        sents.append(raw_sentence)
        except:
            pass
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sents


def delete_stopwords(words):
    stops = set(stopwords.words("russian"))
    words = [stemmer.stem(w) for w in words if w not in stops]
    words = [w for w in words if w and w.isalpha()]
    return words

def process_the_board(imageboard):
    length = 0
    # open last file with parsed sentences
    # sents = [s for s in sents if s] #temp
    unlabeled = pd.read_csv('messages-'+imageboard+'.csv', header=0, delimiter="\t", quoting = csv.QUOTE_MINIMAL, nrows = 2000000)
    print ("Parsing sentences from unlabeled_"+imageboard)
    sents = []
    for message in unlabeled["Text"]:
        sents += message_to_sentences(message, tokenizer, True)
    length = str(int(length) + len(unlabeled))
    print(imageboard + ' board messages count is '+length)
    return sents

# l for pymorphy2 lemmas, s for stemmer, '' for just tokens
# load pymorphy2 dicts
morph = ''
stemmer = ''

if lemmas_bool == 'l':
    morph = pymorphy2.MorphAnalyzer()
elif lemmas_bool == 's':
    stemmer = SnowballStemmer("russian")

# Initialize an empty list of sentences
sentences = []

# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sentences += process_the_board('b')

# already_parsed
# print ("Parsing sentences from specom_b")
# for message in specom_b["Text"]:
#    sentences_b += message_to_sentences(message, tokenizer, True)
# print ("Parsing sentences from specom_p")
# for message in specom_p["Text"]:
#    sentences_p += message_to_sentences(message, tokenizer, True)
# Import the built-in logging module and configure it so that Word2Vec
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

# Set values for various parameters
num_features = 100    # Word vector dimensionality
min_word_count = 4   # Minimum word count
num_workers = 1       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-3   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print ("Training model...")
bigram_transformer = gensim.models.Phrases(sentences)

sentences = bigram_transformer[sentences]

bigram_transformer.save('bigram_transformer_'+ '_'.join(boards)+'_'+str(len(sentences)))
del bigram_transformer
#trigram_transformer = gensim.models.Phrases(sentences)
#sentences = trigram_transformer[sentences]
#del trigram_transformer
model = word2vec.Word2Vec(sentences, workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "%s%dfeatures_%dminwords_%dcontext_bigrams_1e-3" % (lemmas_bool,num_features,min_word_count,context)
model.save(model_name)
