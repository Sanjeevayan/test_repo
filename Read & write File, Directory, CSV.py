1. Python program to read a text file and print it on the screen
import io
from nltk import word_tokenize,sent_tokenize,pos_tag
file = open ("Test4.txt","r")
input_file = file.readline()
sent = sent_tokenize(input_file)
for sentence in sent:
    tokens = word_tokenize(sentence)
    tags = pos_tag(tokens)
    # print (tags)
    for word ,tag in tags:

        print (word,"---->",tag)
file.close()
2. Python program to read a text file and write in a separate file
from nltk import ngrams
file = open ("D:\\Test2.txt","r")
input_file = file.readline()
output_file = open ("D:\\Test4.txt","w")
n = 2
bigrams = ngrams(input_file.split(), n)
for grams in bigrams:
    print (grams)
    output_file.write(str (grams)+ "\n")
file.close()
output_file.close()

from nltk import word_tokenize,sent_tokenize
file = open("D://Test2.txt","r")
input_file = file.readline()
output_file = open("D://Test4.txt","w")
sents = sent_tokenize(input_file)
for sentence in sents:
    tokens = word_tokenize(sentence)
    print (tokens)
    output_file.write(str(tokens) +"\n")
file.close()
output_file.close()


from nltk import sent_tokenize,word_tokenize,pos_tag
file = open("D://Test2.txt","r")
input_file = file.readline()
output_file = open ("D://Test4.txt","w")
sents = sent_tokenize(input_file)
for sentence in sents:
    tokens = word_tokenize(sentence)
    pos = pos_tag(tokens)
    print (pos)
    output_file.write(str(pos)+ "\n")
file.close()
output_file.close()

# Python program to read a Directory of text files and store the output in a text file
# Remarks : working code

import glob
from nltk import word_tokenize,pos_tag
path = "D:\Test\*.txt"
files = glob.glob(path)
K = []
for file in files:
    f = open (file,"r")
    f.readlines()
    f.close()
    K.append(file)
output_file = open ("D:\\Test5.txt","w")
for file in K:
    f = open(file,"r")
    lines = f.readlines()
    for line in lines:
        tokens = word_tokenize(line)
        tags = pos_tag (tokens)
        print (tags)
        output_file.write(str (tags)+ "\n")

output_file.close()

import glob
import nltk

path = "D:\\Test\*.txt"
files = glob.glob(path)
K = [] # creating an array of files
for file in files:
    f=open(file, 'r')
    f.readlines()
    f.close()
    K.append(file)
print (K)

# creating a function to call "Tokenization" Method


def _tokenize_word(input_text):
    tokens = nltk.word_tokenize(input_text)
    return tokens

output_file = open("D:\\Test5.txt", "w")
for file in K:
    f1 = open(file)
    lines = f1.readlines()
    for line in lines:
        result = _tokenize_word(line)
        print (result)
        output_file.write(str(result)+ "\n")
# f1.close()
output_file.close()

# Python program to read a CSV file, run Tokenizer & PoS tagger
# Remarks : working code
import nltk,csv,numpy
from nltk import sent_tokenize, word_tokenize, pos_tag
# reader = csv.reader(open(r'D:\\input_data.csv', 'r'), delimiter= ",",quotechar='|')
reader = csv.reader(open(r'D:\\input_data.csv', 'r'))
for line in reader:
    for field in line:
        tokens = word_tokenize(field)
        tags = pos_tag(tokens)
        print (tokens)
        print (tags)

*********************
from textblob.classifiers import NaiveBayesClassifier as NBC
from textblob import TextBlob
import csv

training_corpus = []
with open('D:\\Train_sentiment_data.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        training_corpus.append(row)

        print(row['sentence'], row['polarity'])
print ("Training Corpus is: ", training_corpus)

test_corpus = []
with open('D:\\Test_sentiment_data.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        test_corpus.append(row)
        # print(row['sentence'], row['polarity'])

print ("Test Corpus is:",test_corpus)


train_data = []
train_labels = []
for row in training_corpus:
    train_data.append(row['sentence'])
    train_labels.append(row['polarity'])
print (train_data)
test_data = []
test_labels = []
for row in test_corpus:
    test_data.append(row['sentence'])
    test_labels.append(row['polarity'])
model = NBC(training_corpus)
print(model.classify("Their codes are amazing."))
print(model.classify("I don't like their computer."))
print(model.accuracy(test_corpus))

*******************************

# Create feature vectors
# vectorizer = TfidfVectorizer(min_df=4, max_df=0.9)
vectorizer = TfidfVectorizer() #(min_df=1, max_df=1) # Arun's suggestion
# print (vectorizer)
# Train the feature vectors
train_vectors = vectorizer.fit_transform(train_data)

# Apply model on test data
test_vectors = vectorizer.transform(test_data)

# Perform classification with SVM, kernel=linear
model = svm.SVC(kernel='linear')
model.fit(train_vectors, train_labels)
prediction = model.predict(test_vectors)
print(prediction)
print (classification_report(test_labels, prediction))

#Validation phase :on new Data set (input: sentences, output: sentences)

validation_text = [ "I love this place"]

new_vector = vectorizer.transform(validation_text)
new_pred = model.predict(new_vector)
print (new_pred)

# Reading and Writing into a CSV file
import nltk,csv,numpy
from nltk import sent_tokenize, word_tokenize, pos_tag
# reader = csv.reader(open(r'D:\\input_data.csv', 'r'), delimiter= ",",quotechar='|')
reader = csv.reader(open(r'D:\\input_data.csv', 'r'))
T = []
for line in reader:
    for field in line:
        tokens = word_tokenize(field)
        tags = pos_tag(tokens)
        print (field)
#         print (tokens)
#         # print (tokens,tags)
#         T.append(tags)
# print (T)
myFile = open('D:\\input_data1.csv','w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(T)

print("Writing complete")

#*********************************

# Writing in a CSV file
import csv

myData = [['First Name', "'Second Name'", "Email","Age"],
          ['Alex', 'Brian', 'A.com',44],
          ['Tom', 'Smith', 'B.com',34],
          ['Donald', 'Trump', 'D.com', 64],]

myFile = open('D:\\input_data.csv','w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerows(myData)

print("Writing complete")

