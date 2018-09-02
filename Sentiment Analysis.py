# **************Code to calculate overall sentiment of a document

#Remarks : working. Need to find overall score/polarity for the whole document

from nltk import sent_tokenize,word_tokenize,pos_tag

positive_words = ["good","excellent", "well","great"]
negative_words = ["bad", "terrible", "shock"]
sentiment_tag = ["JJ","RB"]

input_text = "He is a bad man . He plays well. He is great. "
sents = sent_tokenize(input_text)
# print (sents)
pos_sentiment = []
neg_sentiment = []
for sentence in sents:
    sent_tokens = word_tokenize(sentence)
    pos = pos_tag(sent_tokens)
    print (pos)
    for tag in pos:
        if tag[1] in sentiment_tag:
            print (tag[1],"It has Sentiment",sentence)
            if tag[0] in positive_words:
                pos_sentiment.append(sentence)
                print("POS:positive ", sentence)
            if tag[0]in negative_words:
                print ("POS:negative ", sentence)
                neg_sentiment.append(sentence)
        else:
            print (tag[1],"it doesnot have sentiment")

print ("Positive sentences are:",pos_sentiment)

print ("Negative sentences are:",neg_sentiment)
print (len(pos_sentiment))
print (len(neg_sentiment))
if len(pos_sentiment) > len(neg_sentiment):
    print ("Text is positive ")
else :
    print ("Text is Negative")

#1: *******************************Sentiment Analysis code for a single sentence
import nltk
from nltk import word_tokenize,pos_tag
positive_words = ["good","excellent", "well","great"]
negative_words = ["bad", "terrible", "shock"]
input_text = "He is a bad man."
sent = nltk.sent_tokenize(input_text)
tokens = word_tokenize(input_text)
print (tokens)

for token in tokens:
    if token in positive_words:
        print("Positive")
    elif token in negative_words:
        print ("Negative")
    else :
        print ("Neutral")


#2: ******************* Sentiment Analysis code for multiple sentences

from nltk import sent_tokenize,word_tokenize
positive_words = ["good","excellent", "well","great"]
negative_words = ["bad", "terrible", "shock"]
input_text = "He is a good man. He plays well."
sents = sent_tokenize(input_text)
# tokens = word_tokenize(input_text)

for sentence in sents:
    sent_tokens = word_tokenize(sentence)
    for token in sent_tokens:
      if token in positive_words:
          print ("Token -->",token,"-->","is Positive")
      elif token in negative_words:
          print ("Token -->",token,"-->"," is Negative")
      else:
          print ("Token -->",token,"-->"," is Neutral")

#**********************************

from nltk import sent_tokenize,word_tokenize,pos_tag

sent_dict = ["good","excellent", "well","great"]

input_text = "He is a bad man . He plays well. He is sleeping. "
sents = sent_tokenize(input_text)

for sentence in sents:
    sent_tokens = word_tokenize(sentence)

    sentHasSentiment = False

    for token in sent_tokens:
        if token in sent_dict:
            sentHasSentiment = True

    if (sentHasSentiment):
        print(sentence, "--->It has sentiment")
    else:
        print(sentence, "--->It does not have sentiment")

#*******************https://pythonspot.com/python-sentiment-analysis/

import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import names


def word_feats(words):
    return dict([(word, True) for word in words])


positive_vocab = ['awesome', 'outstanding', 'fantastic', 'terrific', 'good', 'nice', 'great', ':)']
negative_vocab = ['bad', 'terrible', 'useless', 'hate', ':(']
neutral_vocab = ['movie', 'the', 'sound', 'was', 'is', 'actors', 'did', 'know', 'words', 'not']



positive_features = [(word_feats(pos), 'pos') for pos in positive_vocab]
negative_features = [(word_feats(neg), 'neg') for neg in negative_vocab]
neutral_features = [(word_feats(neu), 'neu') for neu in neutral_vocab]

train_set = negative_features + positive_features + neutral_features

classifier = NaiveBayesClassifier.train(train_set)

# Predict
neg = 0
pos = 0
sentence = "Awesome movie, I liked it"
sentence = sentence.lower()
words = sentence.split(' ')

for word in words:
    classResult = classifier.classify(word_feats(word))
    print (word,"--",classResult)
    if classResult == 'neg':
        neg = neg + 1
        # print ("-",neg)
    if classResult == 'pos':
        pos = pos + 1
        print(pos)

print('Positive: ' + str(float(pos) / len(words)))
print('Negative: ' + str(float(neg) / len(words)))