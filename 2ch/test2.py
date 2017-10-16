# -*- coding: UTF-8 -*-
from __future__ import division
import gensim
import re
import csv
import pymorphy2
import nltk.data
import pandas as pd
from nltk.tokenize import WordPunctTokenizer
from nltk.corpus import stopwords
from sklearn.ensemble import (RandomForestClassifier, ExtraTreesClassifier,
                              AdaBoostClassifier)
from nltk.stem.snowball import SnowballStemmer
from nltk import FreqDist
import matplotlib.pyplot as plt
import numpy as np

model = gensim.models.Word2Vec.load('100features_4minwords_10context_bigrams_1e-3')
model_words = set(model.index2word)
stemmer = SnowballStemmer("russian")
bigram_transformer = gensim.models.Phrases.load('bigram_transformer_1')
obscene_list = [u'иди_нахуй', u'заткнись', u'мудак', u'чмо', u'пидор']
error_words = []
# Load the punkt tokenizer
tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

morph = pymorphy2.MorphAnalyzer()
f = open('specom-positive')
r_pos = list(set(f.readlines()))
f.close()

f = open('specom-negative')
r_neg = list(set(f.readlines()))
f.close()

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
        words = [w for w in words if w not in stops]
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
    lemmas = [l for l in lemmas if l.isalpha()]
    lemmas = bigram_transformer[lemmas]
    return(lemmas)
    # return(words)


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
                scores.append(model.n_similarity([word], obscene_list))
            else:
                error_words.append(word)
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


def scrap_messages(mess):
    sentences = []
    for line in mess:
        message = re.findall('(<.*?>)(.*?)(<.*)',line)
        if message:
            message = message[0][1]
            sentences += message_to_sentences(message, tokenizer, '', True)
    return sentences

sentences_pos = scrap_messages(r_pos)
sentences_neg = scrap_messages(r_neg)

sent_pos_vals = [train_classifier(m) for m in sentences_pos]
sent_neg_vals = [train_classifier(m) for m in sentences_neg]

print('random_forest')
sent_pos_slice = int(round(len(sent_pos_vals)*0.9))
sent_neg_slice = int(round(len(sent_neg_vals)*0.9))
train = sent_pos_vals[:sent_pos_slice] + sent_neg_vals[:sent_neg_slice]
train_values = []
test = sent_pos_vals[sent_pos_slice:] + sent_neg_vals[sent_neg_slice:]
test_sents = sent_pos_vals[sent_pos_slice:] + sent_neg_vals[sent_neg_slice:]
target = [0] * sent_pos_slice + [1] * sent_neg_slice
test_res = [0]*(len(sent_pos_vals) - sent_pos_slice) + [1]*(len(sent_neg_vals) - sent_neg_slice)
rf = RandomForestClassifier(n_estimators = 100, n_jobs=2)
print('training random forest')
train = pd.DataFrame(train)
train = train.fillna(0)
rf.fit(train, target)
test = pd.DataFrame(test)
test = test.fillna(0)
prediction = rf.predict(test)
score = 0
for i in range(len(prediction)-1):
    if prediction[i] == test_res[i]:
        score +=1
print ('rf result is ', 100*score/len(prediction))
print rf.feature_importances_


unlabeled = pd.read_csv('messages-b.csv', header=0, delimiter="\t", quoting = csv.QUOTE_MINIMAL)
unlabeled = unlabeled[:10000]

predict = unlabeled["Text"] # messages
predict = [message_to_sentences(m, tokenizer, '', True) for m in predict]
predict_scores = [train_classifier(m) for m in predict]
predict_scores = pd.DataFrame(predict_scores)
predict_scores = predict_scores.fillna(0)
prediction = rf.predict(predict_scores)
tags_pos = []
tags_neg = []

def process_tags(tags):
    tags = [str(t) for t in tags]
    tags = [t.split(',') for t in tags]
    tags_new = []
    for t in tags:
        if len(t)>2:
            tags_new.append(', '.join(t[:2]))
        elif len(t) == 2:
            tags_new.append(', '.join(t))
        else:
            tags_new.append(str(t[0]))
    tags_new = [t.split(' ') for t in tags_new]
    tags = tags_new
    tags = FreqDist(tags)
    t_sum = sum(tags.values())
    tags = [tag for tag in tags.most_common() if 'UNKN' not in tag[0] and 'LATN' not in tag[0] and 'PNCT' not in tag[0]]
    tags = [t for t in tags if 'CONJ' not in t[0] and 'PREP' not in t[0] and 'PRCL' not in t[0] ]
    return tags, t_sum

for i in range(len(prediction)- 1):
    if i%1000 == 0:
        print i
    if prediction[i] == 1:
        for sent in predict[i]:
            tags_pos += [morph.parse(w)[0].tag for w in sent]
    else:
        for sent in predict[i]:
            tags_neg += [morph.parse(w)[0].tag for w in sent]

tags_pos, pos_sum = process_tags(tags_pos)
tags_neg, neg_sum = process_tags(tags_neg)
x = [i for i in range(0,10)]
y1 = zip(*tags_pos[:10])[1]
y1 = [round(100*el/pos_sum,2) for el in y1]
labels1 = zip(*tags_pos[:10])[0]
labels2 = zip(*tags_pos[:200])[0]
y2_val = zip(*tags_neg[:100])[1]
y2 = []
for i in range(len(labels2)-1):
    if labels2[i] in labels1:
        y2.append(y2_val[i])
y2 = [round(100*el/neg_sum,2) for el in y2]
plt.xticks(np.arange(len(labels)),labels, rotation = 90)
plt.plot(x,y1,'-',x, y2, 'r--')
plt.xlabel('Tags')
plt.ylabel('Percentage (%)')
plt.title('')
#plt.yscale('log', linthreshy=0.12)
plt.show()

