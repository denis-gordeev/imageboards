# -*- coding: UTF-8 -*-
import pandas as pd
# Import various modules for string cleaning
from bs4 import BeautifulSoup
import re
import nltk.data
from nltk.corpus import stopwords
from nltk.tokenize import WordPunctTokenizer
from nltk.stem.snowball import SnowballStemmer
from gensim.models import word2vec
import gensim
import csv
import pymorphy2
import cPickle
#l for pymorphy2 lemmas, s for stemmer, '' for just tokens
lemmas_bool = ''
#load pymorphy2 dicts
morph = ''
stemmer = ''
if lemmas_bool == 'l':
    morph = pymorphy2.MorphAnalyzer()
elif lemmas_bool == 's':
    stemmer = SnowballStemmer("english")
boards = ['b', 'pol']
# Read data from files
unlabeled_b = pd.read_csv( 'messages-b.csv', header=0,
 delimiter="\t", quoting = csv.QUOTE_MINIMAL, error_bad_lines=False )
#test = pd.read_csv( "testData.tsv", header=0, delimiter="\t", quoting=3 )
unlabeled_p = pd.read_csv( "messages-pol.csv", header=0,
 delimiter="\t", quoting = csv.QUOTE_MINIMAL, error_bad_lines=False )

#already_parsed
#specom_b = pd.read_csv( 'specom-b.csv', header=0,
#delimiter="\t", quoting = csv.QUOTE_MINIMAL )
#specom_p = pd.read_csv( "specom-po.csv", header=0,
#delimiter="\t", quoting = csv.QUOTE_MINIMAL )


def message_to_wordlist(message, lemmas_bool, remove_stopwords=False):
    # Function to convert a document to a sequence of words,
    # optionally removing stop words.  Returns a list of words.
    #
    # 1. Remove HTML
    #review_text = BeautifulSoup(review).get_text()
    #
    # 2. Remove messages numbers
    message_text = re.sub(">>\d+","", message)
    message_text = message_text.lower()
    message_text = re.sub(u"ё", 'e', message_text, re.UNICODE)
    tokenizer = WordPunctTokenizer()
    # 3. Convert words to lower case and split them
    words = tokenizer.tokenize(message_text)
    lemmas = []
    # 4. Optionally remove stop words (false by default)
    if remove_stopwords:
        stops = set(stopwords.words("english"))
        words = [w for w in words if not w in stops]
    if lemmas_bool == 'l':
        for word in words:
            word_parsed = morph.parse(word)
            if len(word_parsed) > 0:
                lemmas.append(word_parsed[0].normal_form)
    elif lemmas_bool == 's':
        for word in words:
            word = stemmer.stem(word)
            if len(word) > 0:
                lemmas.append(word)
    else:
        lemmas = words
    # 5. Return a list of words
    return(lemmas)
    #return(words)

# Define a function to split a message into parsed sentences
def message_to_sentences( message, tokenizer, lemmas_bool, remove_stopwords=False):
    sentences = []
    # Function to split a message into parsed sentences. Returns a
    # list of sentences, where each sentence is a list of words
    #
    # 1. Use the NLTK tokenizer to split the paragraph into sentences
    if type(message) == str:
        message = message.decode('utf-8')
        raw_sentences = tokenizer.tokenize(message.strip())
        #
        # 2. Loop over each sentence
        for raw_sentence in raw_sentences:
            # If a sentence is empty, skip it
            if len(raw_sentence) > 0:
                # Otherwise, call message_to_wordlist to get a list of words
                sentences.append( message_to_wordlist( raw_sentence,lemmas_bool, remove_stopwords))
    #
    # Return the list of sentences (each sentence is a list of words,
    # so this returns a list of lists
    return sentences

def process_the_board(imageboard):
    length = 0
    # open last file with parsed sentences
    objects = os.listdir(".")
    length = max([re.findall('\d+', f)[0] for f in objects if f.startswith(lemmas_bool+'_sentences_'+imageboard)])
    filename = lemmas_bool+'_sentences_'+imageboard+length+'.pickle'
    print ('Opening file ' + filename)
    f = open(filename, 'rb')
    sents = cPickle.load(f)
    f.close()
    # sents = [delete_stopwords(s) for s in sents] #temp
    # sents = [s for s in sents if s] #temp
    print ("Loaded sentences for "+imageboard + ' ' + str(len(sents)))
    unlabeled = pd.read_csv('messages-'+imageboard+'.csv', header=0, delimiter="\t", quoting = csv.QUOTE_MINIMAL, skiprows=range(1, int(length)+1))
    print ("Parsing sentences from unlabeled_"+imageboard)
    for message in unlabeled["Text"]:
        sents += message_to_sentences(message, tokenizer, True)
    length = str(int(length) + len(unlabeled))
    print(imageboard + ' board messages count is '+length)
    cPickle.dump(sents, open(lemmas_bool + '_sentences_' + imageboard + length + '.pickle', 'wb'))
    return sents


# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

sentences = []  # Initialize an empty list of sentences
sentences_b = []
sentences_p = []
#print ("Parsing sentences from specom_b")
#for message in specom_b["Text"]:
#    sentences_b += message_to_sentences(message, tokenizer, True)
#print ("Parsing sentences from specom_p")
#for message in specom_p["Text"]:
#    sentences_p += message_to_sentences(message, tokenizer, True)

print ("Parsing sentences from unlabeled_pol")
for message in unlabeled_p["Text"]:
    sentences_p += message_to_sentences(message, tokenizer, True)

print ("Parsing sentences from unlabeled_b")
for message in unlabeled_b["Text"]:
    sentences_b += message_to_sentences(message, tokenizer, True)


cPickle.dump(sentences_b, open(lemmas_bool + '_sentences_b'+str(len(unlabeled_b)) +'.pickle', 'wb'))
cPickle.dump(sentences_p, open(lemmas_bool + '_sentences_pol'+str(len(unlabeled_p)) +'.pickle', 'wb'))
sentences = sentences_b + sentences_p
# Import the built-in logging module and configure it so that Word2Vec
# creates nice output messages
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s',\
    level=logging.INFO)

# Set values for various parameters
num_features = 600    # Word vector dimensionality
min_word_count = 4   # Minimum word count
num_workers = 1       # Number of threads to run in parallel
context = 10          # Context window size
downsampling = 1e-4   # Downsample setting for frequent words

# Initialize and train the model (this will take some time)
print ("Training model...")
bigram_transformer = gensim.models.Phrases(sentences)
bigram_transformer.save('bigram_transformer_'+ '_'.join(boards)+'_'+str(len(sentences)))
model = word2vec.Word2Vec(bigram_transformer[sentences], workers=num_workers, \
            size=num_features, min_count = min_word_count, \
            window = context, sample = downsampling)

# If you don't plan to train the model any further, calling
# init_sims will make the model much more memory-efficient.
model.init_sims(replace=True)

# It can be helpful to create a meaningful model name and
# save the model for later use. You can load it later using Word2Vec.load()
model_name = "%s%dfeatures_%dminwords_%dcontext_bigrams_1e-3" % (lemmas_bool,num_features,min_word_count,context)
model.save(model_name)
