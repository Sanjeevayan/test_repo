from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import metrics
from sklearn.naive_bayes import MultinomialNB
import csv
import pandas as pd
training_corpus = []
with open('D:\\Train_sentiment_data.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
       training_corpus.append(row)

#        print(row['sentence'], row['polarity'])
# print ("Training Corpus is: ", training_corpus)

test_corpus = []
with open('D:\\Test_sentiment_data.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        test_corpus.append(row)
        # print(row['sentence'], row['polarity'])

# print ("Test Corpus is:",test_corpus)

# preparing data for SVM model
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
# Create feature vectors
# vectorizer = TfidfVectorizer(min_df=4, max_df=0.9)
vectorizer = TfidfVectorizer() #(min_df=1, max_df=1) # Arun's suggestion
# vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1,3), min_df = 0, stop_words = 'english') # Sanjeev's suggestion
# print (vectorizer)
# Train the feature vectors
train_vectors = vectorizer.fit_transform(train_data)
X_train = vectorizer.transform(train_data)
print(X_train.shape) # gives the dimension of the matrix
print (vectorizer.vocabulary_)#gives the vocabulary size
# Apply model on test data
test_vectors = vectorizer.transform(test_data)

# Perform classification with SVM, kernel=linear
model = MultinomialNB()
model.fit(train_vectors, train_labels)
prediction = model.predict(test_vectors)
print(prediction)
print (classification_report(test_labels, prediction))
print(metrics.confusion_matrix(test_labels, prediction)) # Confusion Matrix


# Validation phase :on new Data set (input: sentences, output: sentences)

validation_text = [ "I love this place"]

new_vector = vectorizer.transform(validation_text)

new_pred = model.predict(new_vector)
print (new_pred)

# Now I want to test classification on multiple sentence of a list. Each sentence should get a "class tag"
validation_text =  ["I do not love this place.","I do not hate this hotel.","I like fruits."]
sent_validation_text = validation_text #sent_tokenize(str(validation_text))
# print (sent_validation_text)
#for sent in sent_validation_text:
    # print (sent,new_vector)

new_vector = vectorizer.transform(sent_validation_text)

new_pred = model.predict(new_vector)

print(list(zip(sent_validation_text, new_pred)))

