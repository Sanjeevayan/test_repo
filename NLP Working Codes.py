# *********** Code for SVM classiifer using SK learn and reading a csv file
# Remarks : working code

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm
import csv

training_corpus = []
with open('D:\\Train_sentiment_data.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        training_corpus.append(row)

        print(row['sentence'], row['polarity'])
print("Training Corpus is: ", training_corpus)

test_corpus = []
with open('D:\\Test_sentiment_data.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        test_corpus.append(row)
        # print(row['sentence'], row['polarity'])

print("Test Corpus is:", test_corpus)

# preparing data for SVM model
train_data = []
train_labels = []
for row in training_corpus:
    train_data.append(row['sentence'])
    train_labels.append(row['polarity'])
print(train_data)
test_data = []
test_labels = []
for row in test_corpus:
    test_data.append(row['sentence'])
    test_labels.append(row['polarity'])
# Create feature vectors
vectorizer = TfidfVectorizer(min_df=4, max_df=0.9)
# print (vectorizer)
# Train the feature vectors
train_vectors = vectorizer.fit_transform(train_data)
# Apply model on test data
test_vectors = vectorizer.transform(test_data)

# Perform classification with SVM, kernel=linear
model = svm.SVC(kernel='linear')
model.fit(train_vectors, train_labels)
prediction = model.predict(test_vectors)
# print(prediction)
print(classification_report(test_labels, prediction))

#Validation on new Data set


validation_text = [ "I love this place"]


new_vector = vectorizer.transform(validation_text)
new_pred = model.predict(new_vector)
print (new_pred)

# Python code to read a txt file and store the output in another txt file. Remarks: working code
from nltk import word_tokenize,pos_tag
file = open ("D:\\Test2.txt","r")
input_file = file.readline()
output_file = open("D:\\Test4.txt","w")
tokens = word_tokenize(input_file)
pos = pos_tag(tokens)
K = []
for tag in pos:
    K.append (tag)
print (K)
output_file.write(str(K)+ "\n")
file.close()
output_file.close()

#*********************************

import nltk
from collections import Counter
# input = "The CBI will submit a chargesheet against Allahabad Bank CEO & MD Usha Ananthasubramanian and several others"

# File = open(fileName) #open file
# lines = File.read() #read all lines

file = open ("D:\\Test1.txt","r")
input_file = file.readline()
output_file = open("D:\\Test4.txt","w")
sentences = nltk.sent_tokenize(input_file) #tokenize sentences
nouns = [] #empty to array to hold all nouns

for sentence in sentences:
     for token in nltk.word_tokenize(str(sentence)): # understand this sentence/syntax
         for word, pos in nltk.pos_tag(token):
             if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
                    nouns.append(word)
                    output_file.write(word + "\n")
file.close()
output_file.close()

# How to read all the files of the folder and write in : 1) single file 2) single folder
import sys
import glob
from nltk import ngrams
file = open (r"D:\Test\Trial1.txt","r")
# file = open (r"D:\Test\*.txt","r")
output_file = open("D:\\Trial4.txt", "w")
n = 2

for file in glob.iglob(r"D:\Test\*.txt"):
    input_file = open(file).read() #OSError: [Errno 22] Invalid argument: 'D:\\Test\\*.txt'
    bigrams = ngrams(input_file.split(), n)
    for grams in bigrams:
        # print (grams)
        output_file.write(str (grams)+ "\n")
#    input_file.close()
output_file.close()

#1*****************************Keyword Extraction using NLTK PoS Tagger with frequency: Approach:1************************
import nltk
from collections import Counter
input = "The CBI will submit a chargesheet against Allahabad Bank CEO & MD Usha Ananthasubramanian and several others"

# File = open(fileName) #open file
# lines = File.read() #read all lines
sentences = nltk.sent_tokenize(input) #tokenize sentences
nouns = [] #empty to array to hold all nouns

for sentence in sentences:
     for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))):
         if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
             nouns.append(word)
print (nouns)

frequency = Counter (nouns)
print (frequency)

#2*********************** Keyword Extraction using TextBlob with frequency:Approach2************************
from textblob import TextBlob
from collections import Counter
input = "The CBI will submit a chargesheet against  Usha Ananthasubramanian and several others. CBI will arrest Usha Ananthasubramanian in May "
blob = TextBlob(input)
phrases = blob.noun_phrases

frequency = Counter (phrases)
print (frequency)

#3*********************** Problem Statement1: Filtering of a PoS Tag & Frequency of PoS Tags***********************

# option1
from nltk import word_tokenize, pos_tag
from collections import Counter

text = "I am Sanjeev. I stay in Bangalore"
tokens = word_tokenize(text)
tags = pos_tag(tokens)
d = dict()
for token, tag in tags:
    d[token] = tag
print(d)

# option2

from nltk import word_tokenize,pos_tag
from collections import Counter
text = "I am Sanjeev. I stay in Bangalore"
tokens = word_tokenize(text)
tags = pos_tag(tokens)

for token,tag in tags:
    X = token ,"--->", tag
    print ("Tags of the tokens are",X) # prints all the tokens alonwith corresponding POS tags.

count = Counter([(token,tag) for token,tag in tags])
# count = Counter([token for token,tag in tags])
# count = Counter([tag for token,tag in tags])

# Trying to filter only a particular tag i.e NNP

for item in count:
    if "NNP" in item[1]:
        print ("NNPs are",item)

print ("Term Frequencies are:",count)
#4 Code for generating n gram (Remarks : able to read the file but unable to write it)


from nltk import ngrams
#sentence = 'this is a foo bar sentences and i want to ngramize it'
file = open ("D:\\Test2.txt","r")
input_file = file.readline()
output_file = open ("D:\\Test4.txt","w")
n = 2
bigrams = ngrams(input_file.split(), n)
for grams in bigrams:
    print (grams)
#5***********************Working codes***************

#*******Remarks: working code

from nltk import ngrams
file = open ("D:\\Test2.txt","r")
input_file = file.readline()
output_file = open ("D:\\Test4.txt","w")
n = 2
bigrams = ngrams(input_file.split(), n)
for grams in bigrams:
    # print (grams)
    output_file.write(str (grams)+ "\n")
file.close()
output_file.close()


# Doubt1: How to read all the files of the folder and write in : 1) single file 2) single folder
#Remarks: working for a single file
import sys
import glob
from nltk import ngrams
file = open (r"D:\Test\Trial1.txt","r")
# file = open (r"D:\Test\*.txt","r")
input_file = file.readline() #OSError: [Errno 22] Invalid argument: 'D:\\Test\\*.txt'
output_file = open ("D:\\Trial4.txt","w")
n = 2
bigrams = ngrams(input_file.split(), n)
for grams in bigrams:
    # print (grams)
    output_file.write(str (grams)+ "\n")
file.close()
output_file.close()

#************** Print NER Tags alongwith its Frequencies https://www.commonlounge.com/discussion/2662a77ddcde4102a16d5eb6fa2eff1e
import nltk
from collections import Counter

doc = '''Andrew Yan-Tak Ng is a Chinese American computer scientist.
He is the former chief scientist at Baidu, where he led the company's
Artificial Intelligence Group. He is an adjunct professor (formerly 
associate professor) at Stanford University. Ng is also the co-founder
and chairman at Coursera, an online education platform. Andrew was born
in the UK in 1976. His parents were both from Hong Kong.'''
# tokenize doc
tokenized_doc = nltk.word_tokenize(doc)

# tag sentences and use nltk's Named Entity Chunker
tagged_sentences = nltk.pos_tag(tokenized_doc)
ne_chunked_sents = nltk.ne_chunk(tagged_sentences)

# extract all named entities
named_entities = []
for tagged_tree in ne_chunked_sents:
    if hasattr(tagged_tree, 'label'):
        entity_name = ' '.join(c[0] for c in tagged_tree.leaves())  #
        entity_type = tagged_tree.label()  # get NE category
        named_entities.append((entity_name, entity_type))
print(named_entities)
count = Counter (named_entities)
print (count)
#6*********************Tf-Idf using Textblob***************
import math
from textblob import TextBlob as tb

def tf(word, blob):
    return blob.words.count(word) / len(blob.words)

def n_containing(word, bloblist):
    return sum(1 for blob in bloblist if word in blob.words)

def idf(word, bloblist):
    return math.log(len(bloblist) / (1 + n_containing(word, bloblist)))

def tfidf(word, blob, bloblist):
    return tf(word, blob) * idf(word, bloblist)

document1 = tb("""Python is a 2000 made-for-TV horror movie directed by Richard
Clabaugh. The film features several cult favorite actors, including William
Zabka of The Karate Kid fame, Wil Wheaton, Casper Van Dien, Jenny McCarthy,
Keith Coogan, Robert Englund (best known for his role as Freddy Krueger in the
A Nightmare on Elm Street series of films), Dana Barron, David Bowe, and Sean
Whalen. The film concerns a genetically engineered snake, a python, that
escapes and unleashes itself on a small town. It includes the classic final
girl scenario evident in films like Friday the 13th. It was filmed in Los Angeles,
 California and Malibu, California. Python was followed by two sequels: Python
 II (2002) and Boa vs. Python (2004), both also made-for-TV films.""")

document2 = tb("""Python, from the Greek word (πύθων/πύθωνας), is a genus of
nonvenomous pythons[2] found in Africa and Asia. Currently, 7 species are
recognised.[2] A member of this genus, P. reticulatus, is among the longest
snakes known.""")

document3 = tb("""The Colt Python is a .357 Magnum caliber revolver formerly
manufactured by Colt's Manufacturing Company of Hartford, Connecticut.
It is sometimes referred to as a "Combat Magnum".[1] It was first introduced
in 1955, the same year as Smith &amp; Wesson's M29 .44 Magnum. The now discontinued
Colt Python targeted the premium revolver market segment. Some firearm
collectors and writers such as Jeff Cooper, Ian V. Hogg, Chuck Hawks, Leroy
Thompson, Renee Smeets and Martin Dougherty have described the Python as the
finest production revolver ever made.""")

bloblist = [document1, document2, document3]
for i, blob in enumerate(bloblist):
    print("Top words in document {}".format(i + 1))
    scores = {word: tfidf(word, blob, bloblist) for word in blob.words}
    sorted_words = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    for word, score in sorted_words[:3]:
        print("\tWord: {}, TF-IDF: {}".format(word, round(score, 5)))

#****** Code to create Bag of words  https://machinelearningmastery.com/prepare-text-data-machine-learning-scikit-learn/

# SK Learn code for tf-df vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# list of text documents
text = ["The quick brown fox jumped over the lazy dog.",
		"The dog.",
		"The fox"]
# create the transform
vectorizer = TfidfVectorizer()
# tokenize and build vocab
vectorizer.fit(text)
# summarize
print(vectorizer.vocabulary_)
print(vectorizer.idf_)
# encode document 1
vector = vectorizer.transform([text[0]])
# summarize encoded vector
print(vector.shape)
print(vector.toarray())