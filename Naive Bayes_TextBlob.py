from textblob.classifiers import NaiveBayesClassifier as NBC
from textblob import TextBlob
import csv

training_corpus = []
with open('D:\\Train_sentiment_data.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        training_corpus.append(row)

        # print(row['sentence'], row['polarity'])
# print ("Training Corpus is: ", training_corpus)
print (training_corpus)
test_corpus = []
with open('D:\\Test_sentiment_data.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        test_corpus.append(row)
        # print(row['sentence'], row['polarity'])

# print ("Test Corpus is:",test_corpus)


train_data = []
train_labels = []
for row in training_corpus:
    train_data.append(row['sentence'])
    train_labels.append(row['polarity'])
# print (train_data)
test_data = []
test_labels = []
for row in test_corpus:
    test_data.append(row['sentence'])
    test_labels.append(row['polarity'])

model = NBC(training_corpus)
print(model.classify("Their codes are amazing."))
print(model.classify("I don't like their computer."))

print(model.accuracy(test_corpus))
