# -*- coding: UTF-8 -*-
from __future__ import division
import gensim
import re
import nltk.data
import pandas as pd
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from nltk.stem.snowball import SnowballStemmer
import csv
import matplotlib.pyplot as plt
import numpy as np

model = gensim.models.Word2Vec.load('s600features_4minwords_10context_bigrams_1e-3')
model_words = set(model.index2word)
#l for pymorphy2 lemmas, s for stemmer, '' for just tokens
lemmas_bool = 's'
agg_words = ['cunt','shut_up']
morph = ''
stemmer = ''
if lemmas_bool == 'l':
    pass
elif lemmas_bool == 's':
    stemmer = SnowballStemmer("english")
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

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

def sentence_semantic_difference(sentence):
    in_model = [word for word in sentence if word in model_words]
    scores = []
    if len(in_model) < 2:
        return 0
    for i in range(len(in_model)-2):
        for k in range(1,len(in_model)-1):
            scores.append(model.similarity(in_model[i],in_model[k]))
    if scores:
        return sum(scores)/len(scores)
    else:
        return 0

def avg(l):
    if l:
        return sum([len(s) for s in l])/len(l)
    else:
        return 0

def train_classifier(sents):
    sents_scores = []
    for sent in sents:
        scores = []
        res = []
        for word in sent:
            if word in model_words:
                scores.append(model.n_similarity([word], agg_words))
        if scores:
            res+= max(scores)-min(scores), max(scores), min(scores), sum(scores)/len(scores)
        else:
            res+=0,0,0,0
        if sent:
            res += len(sent), max([len(w) for w in sent]), min([len(w) for w in sent]), avg(sent),sentence_semantic_difference(sent)
        else:
            res += 0,0,0,0,0
        sents_scores.append(res)
    sents_scores = [sum(i) for i in zip(*sents_scores)]
    return sents_scores


messages = pd.read_csv( 'aggression.csv', header=0,
 delimiter="\t", quoting = csv.QUOTE_MINIMAL )

delimiter = int(len(messages)*0.9)

train = messages[:delimiter]
train_m = train[:]["Text"] # messages
train_t = list(train[:]["Aggression"]) # target

test = messages[delimiter:]
test_m = test[:]["Text"] # messages
test_t = list(test[:]["Aggression"]) # target

error_words = []
train_m = [message_to_sentences(m, tokenizer, True) for m in train_m]
test_m = [message_to_sentences(m, tokenizer, True) for m in test_m]

train_scores = [train_classifier(m) for m in train_m]
test_scores = [train_classifier(m) for m in test_m]

rf = RandomForestClassifier(n_estimators = 100, n_jobs=2)
print('training random forest')
#train_scores = pd.DataFrame(train_scores)
train = train.fillna(0)
test = test.fillna(0)
rf.fit(train_scores, train_t)
test_scores = pd.DataFrame(test_scores)

prediction = rf.predict(test_scores)
score = 0
for i in range(len(prediction)-1):
    if prediction[i] == test_t[i]:
        score +=1
print ('rf result is ', 100*score/len(prediction))
print rf.feature_importances_


unlabeled = pd.read_csv('messages-b.csv', header=0, delimiter="\t", quoting = csv.QUOTE_MINIMAL)
unlabeled = unlabeled[:10000]

predict = unlabeled["Text"] # messages
predict = [message_to_sentences(m, tokenizer, True) for m in predict]
predict_scores = [train_classifier(m) for m in predict]
predict_scores = pd.DataFrame(predict_scores)
predict_scores = predict_scores.fillna(0)
prediction = rf.predict(predict_scores)

tags_pos = []
tags_neg = []
for i in range(len(prediction)- 1):
    if prediction[i] == 0:
        for sent in predict[i]:
            tags_pos += nltk.pos_tag(sent)
    else:
        for sent in predict[i]:
            tags_neg += nltk.pos_tag(sent)
tags_pos = FreqDist(tags_pos)
tags_neg = FreqDist(tags_neg)
pos_sum = sum(tags_pos.values())
neg_sum = sum(tags_neg.values())
