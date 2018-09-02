#*********** Code for SVM classiifer using SK learn and reading a csv file
#Remarks : working code

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn import svm
import csv
training_corpus = []
with open('D:\\Train_sentiment_data.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        training_corpus.append(row)

#         print(row['sentence'], row['polarity'])
# print ("Training Corpus is: ", training_corpus)

test_corpus = []
with open('D:\\Test_sentiment_data.csv', newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        test_corpus.append(row)
        # print(row['sentence'], row['polarity'])
#
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
# print (classification_report(test_labels, prediction))

#Validation on new Data set

validation_text = [ "Food is great here"]


new_vector = vectorizer.transform(validation_text)
new_pred = model.predict(new_vector)
print (validation_text,"--->",new_pred)

validation_text = ["I love this place","I hate this hotel","I like it"]

# Validation on new Data set wih multiple sentences
for sent in validation_text:
    new_vector = vectorizer.transform(validation_text)
    # print (sent,new_vector)

    for vect in new_vector:
        new_pred = model.predict(vect)
        # K.append (new_pred)
    print(sent,new_pred)