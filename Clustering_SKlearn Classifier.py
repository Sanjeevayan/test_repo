#https://pythonprogramminglanguage.com/kmeans-text-clustering/
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from sklearn.metrics import adjusted_rand_score
#
# documents = ["This little kitty came to play when I was eating at a restaurant.",
#              "Merley has the best squooshy kitten belly.",
#              "Google Translate app is incredible.",
#              "If you open 100 tab in google you get a smiley face.",
#              "Best cat photo I've ever taken.",
#              "Climbing ninja cat.",
#              "Impressed with google map feedback.",
#              "Key promoter extension for Google Chrome."]
#
# vectorizer = TfidfVectorizer(stop_words='english')
# X = vectorizer.fit_transform(documents)
#
# true_k = 2
# model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
# model.fit(X)
#
# print("Top terms per cluster:")
# order_centroids = model.cluster_centers_.argsort()[:, ::-1]
# terms = vectorizer.get_feature_names()
# for i in range(true_k):
#     print("Cluster %d:" % i),
#     for ind in order_centroids[i, :10]:
#         print(' %s' % terms[ind]),
#     # print
#
# print("\n")
# print("Prediction")
#
# Y = vectorizer.transform(["chrome browser to open."])
# prediction = model.predict(Y)
# print(prediction)
#
# Y = vectorizer.transform(["My cat is hungry."])
# prediction = model.predict(Y)
# print(prediction)

#*****************************************

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import KMeans
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
import re
import numpy as np
from collections import Counter
import os

os.chdir('D:\\')

stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


# Cleaning the text sentences so that punctuation marks, stop words & digits are removed
def clean(doc):
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    processed = re.sub(r"\d+", "", normalized)
    y = processed.split()
    return y


print("There are 10 sentences of following three classes on which K-NN classification and K-means clustering" \
      " is performed : \n1. Cricket \n2. Artificial Intelligence \n3. Chemistry")
path = "KNN &  K Means training data.txt"

train_clean_sentences = []
fp = open(path, 'r')
for line in fp:
    line = line.strip()
    cleaned = clean(line)
    cleaned = ' '.join(cleaned)
    train_clean_sentences.append(cleaned)

vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(train_clean_sentences)

# Creating true labels for 30 training sentences
y_train = np.zeros(30)
y_train[10:20] = 1
y_train[20:30] = 2

# Clustering the document with KNN classifier
modelknn = KNeighborsClassifier(n_neighbors=5)
modelknn.fit(X, y_train)

# Clustering the training 30 sentences with K-means technique
modelkmeans = KMeans(n_clusters=3, init='k-means++', max_iter=200, n_init=100)
modelkmeans.fit(X)

test_sentences = ["Chemical compunds are used for preparing bombs based on some reactions", \
                  "Cricket is a boring game where the batsman only enjoys the game", \
                  "Machine learning is an area of Artificial intelligence", \
                  "I like cricket"]

test_clean_sentence = []
for test in test_sentences:
    cleaned_test = clean(test)
    cleaned = ' '.join(cleaned_test)
    cleaned = re.sub(r"\d+", "", cleaned)
    test_clean_sentence.append(cleaned)

Test = vectorizer.transform(test_clean_sentence)

true_test_labels = ['Cricket', 'AI', 'Chemistry']
predicted_labels_knn = modelknn.predict(Test)
predicted_labels_kmeans = modelkmeans.predict(Test)

print("\nBelow 3 sentences will be predicted against the learned nieghbourhood and learned clusters:\n1. ", \
      test_sentences[0], "\n2. ", test_sentences[1], "\n3. ", test_sentences[2])
print("\n-------------------------------PREDICTIONS BY KNN------------------------------------------")
print("\n", test_sentences[0], ":", true_test_labels[np.int(predicted_labels_knn[0])], \
      "\n", test_sentences[1], ":", true_test_labels[np.int(predicted_labels_knn[1])], \
      "\n", test_sentences[2], ":", true_test_labels[np.int(predicted_labels_knn[2])])

print("\n-------------------------------PREDICTIONS BY K-Means--------------------------------------")
print("\nIndex of Cricket cluster : ", Counter(modelkmeans.labels_[0:10]).most_common(1)[0][0])
print("Index of Artificial Intelligence cluster : ", Counter(modelkmeans.labels_[10:20]).most_common(1)[0][0])
print("Index of Chemistry cluster : ", Counter(modelkmeans.labels_[20:30]).most_common(1)[0][0])

print (test_sentences,predicted_labels_kmeans)
print("\n", test_sentences[0], ":", predicted_labels_kmeans[0], \
      "\n", test_sentences[1], ":", predicted_labels_kmeans[1], \
      "\n", test_sentences[2], ":", predicted_labels_kmeans[2], \
      "\n", test_sentences[3], ":", predicted_labels_kmeans[3] )

#******************************************
# import pandas as pd
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import KMeans
# from sklearn.metrics import adjusted_rand_score
# import sys
#
# documents = open (r"D:\KNN &  K Means training data.txt","r")
# vectorizer = TfidfVectorizer(stop_words='english')
# X = vectorizer.fit_transform(list(documents))
#
# true_k = 3
# model = KMeans(n_clusters=true_k, init='k-means++', max_iter=100, n_init=1)
# model.fit(X)
#
# print("Top terms per cluster:")
# order_centroids = model.cluster_centers_.argsort()[:, ::-1]
# terms = vectorizer.get_feature_names()
# for i in range(true_k):
#     print("Cluster %d:" % i),
#     for ind in order_centroids[i, :10]:
#         print(' %s' % terms[ind]),
#     # print
#
# print("\n")
# print("Prediction")
#
# Y = vectorizer.transform(["I play cricket."])
# prediction = model.predict(Y)
# print(prediction)
#
# Y = vectorizer.transform(["I like AI"])
# prediction = model.predict(Y)
# print(prediction)