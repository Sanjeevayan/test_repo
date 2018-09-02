# link: https://pythonprogramminglanguage.com/logistic-regression-spam-filter/
import csv
import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score

df = pd.read_csv('D:\\Train_sentiment_data.csv')
# df = csv.reader(open('D:\\Train_sentiment_data.csv','r'))

X_train_raw, X_test_raw, y_train, y_test = train_test_split(df["sentence"],df["polarity"])

vectorizer = TfidfVectorizer()

X_train = vectorizer.fit_transform(X_train_raw)
# print (X_train.shape)# Predicts the dimension of the Matrix of vectorized features in classification
classifier = LogisticRegression()
classifier.fit(X_train, y_train)

X_test = vectorizer.transform( ['Food was great in this hotel.'] )
predictions = classifier.predict(X_test)
print(predictions)


def readCSV(fileName):
    with open(fileName) as File:
        reader = csv.reader(File, delimiter=',', quotechar=',',
                            quoting=csv.QUOTE_MINIMAL)
        return [r[0] for r in reader]


def writeCSV(fileName, data):
    myFile = open(fileName, 'w', newline='')
    with myFile:
        writer = csv.writer(myFile)
        writer.writerows(data)



validation_text = readCSV('D:\\Test_sentiment_data.csv')

new_vector = vectorizer.transform(validation_text)

new_pred = classifier.predict(new_vector)

output = []
for i, sent in enumerate(validation_text):
    output.append([sent, new_pred[i]])

writeCSV('D:\\output_data.csv', output)

print (output)
