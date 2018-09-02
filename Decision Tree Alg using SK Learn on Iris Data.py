# Option1

from sklearn.datasets import load_iris
from sklearn import tree
import numpy as np
iris = load_iris()
print (iris.feature_names)
print (iris.target_names)
print(iris.data[0])
print(iris.target[0])
removed = [0,50,100]
new_target = np.delete(iris.target,removed)
new_data = np.delete(iris.data,removed,axis = 0)
clf = tree.DecisionTreeClassifier()
clf = clf.fit(new_data,new_target)
prediction = clf.predict(iris.data[removed])
print("Original Labels as per data",iris.target[removed])
print("Predicted Labels as per algorithm",iris.target[removed])

# Option2

# Sample Decision Tree Classifier
from sklearn import datasets
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier
# load the iris datasets
dataset = datasets.load_iris()
# fit a CART model to the data
model = DecisionTreeClassifier()
model.fit(dataset.data, dataset.target)
print(model)
# make predictions
expected = dataset.target
predicted = model.predict(dataset.data)
print (predicted)
# summarize the fit of the model
print(metrics.classification_report(expected, predicted))
print(metrics.confusion_matrix(expected, predicted))

# OPtion3:

from sklearn import tree
clf = tree.DecisionTreeClassifier()

#[height, hair-length, voice-pitch]
X = [ [180, 15,0],
      [167, 42,1],
      [136, 35,1],
      [174, 15,0],
      [141, 28,1]]

Y = ['man', 'woman', 'woman', 'man', 'woman']

clf = clf.fit(X, Y)
prediction = clf.predict([[133, 37,1]])
print(prediction)

#************************************
#Decision Tree on Textual Data

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import classification_report
from sklearn.tree import DecisionTreeClassifier
from nltk import sent_tokenize
training_corpus = [
                   ('I am exhausted of this work.', 'Class_B'),
                   ("I can't cooperate with this", 'Class_B'),
                   ('He is my badest enemy!', 'Class_B'),
                   ('My management is poor.', 'Class_B'),
                   ('I love this burger.', 'Class_A'),
                   ('This is an brilliant place!', 'Class_A'),
                   ('I feel very good about these dates.', 'Class_A'),
                   ('This is my best work.', 'Class_A'),
                   ("What an awesome view", 'Class_A'),
                   ('I do not like this dish', 'Class_B')]
test_corpus = [
                ("I am not feeling well today.", 'Class_B'),
                ("I feel brilliant!", 'Class_A'),
                ('Gary is a friend of mine.', 'Class_A'),
                ("I can't believe I'm doing this.", 'Class_B'),
                ('The date was good.', 'Class_A'), ('I do not enjoy my job', 'Class_B')]
# preparing data for SVM model
train_data = []
train_labels = []
for row in training_corpus:
    train_data.append(row[0])
    train_labels.append(row[1])

test_data = []
test_labels = []
for row in test_corpus:
    test_data.append(row[0])
    test_labels.append(row[1])

# Create feature vectors
vectorizer = TfidfVectorizer(min_df=4, max_df=0.9)
# Train the feature vectors
train_vectors = vectorizer.fit_transform(train_data)
# Apply model on test data
test_vectors = vectorizer.transform(test_data)

# Perform classification with SVM, kernel=linear
model = DecisionTreeClassifier()
model.fit(train_vectors, train_labels)
prediction = model.predict(test_vectors)
print(prediction)
print (classification_report(test_labels, prediction))

validation_text = [ "I love this place"]

new_vector = vectorizer.transform(validation_text)
new_pred = model.predict(new_vector)
print (validation_text,"--->",new_pred)