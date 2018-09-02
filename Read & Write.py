#*******Remarks: working code

# from nltk import ngrams
# file = open ("D:\\Test2.txt","r")
# input_file = file.readline()
# output_file = open ("D:\\Test4.txt","w")
# n = 2
# bigrams = ngrams(input_file.split(), n)
# for grams in bigrams:
#     # print (grams)
#     output_file.write(str (grams)+ "\n")
# file.close()
# output_file.close()
#
# # *******Remarks: working code
# from nltk import sent_tokenize
# file = open ("D:\\Test2.txt","r")
# input_file = file.readline()
# output_file = open ("D:\\Test4.txt","w")
# K = []
# for line in input_file:
#     sent = sent_tokenize(input_file)
#     K.append(sent)
# print (K)
# output_file.write(K + "\n")
# file.close()
# output_file.close()

# # Program to read multiple files from a folder and store the output in a single file
# # Remarks = working code
# import sys
# import glob
# from nltk import ngrams
# # file = open (r"D:\Test\Trial1.txt","r")
# # file = open (r"D:\Test\*.txt","r")
# # output_file = open("D:\\Trial4.txt", "w")
# n = 2
# K = []
# output_file = open("D:\\Trial4.txt", "w")
# for file in glob.iglob(r"D:\Test\*.txt"):
#     input_file = open(file).read()
#
#     bigrams = ngrams(input_file.split(), n)
#     for grams in bigrams:
#         K.append (grams)
#
# print (K)
# output_file.write(str (K)+ "\n")
#
# output_file.close()

# import glob
# from nltk import ngrams
#
#
# file = open (r"D:\Test\Trial1.txt","r")
# output_file = open("D:\\Trial4.txt", "w")
# n = 2
#
#
# for file in glob.iglob(r"D:\Test\Trial1.txt"):
#     input_file = open(file).read() #OSError: [Errno 22] Invalid argument: 'D:\\Test\\*.txt'
#     bigrams = ngrams(input_file.split(), n)
#     for grams in bigrams:
#         print (grams)
#         output_file.write(str (grams)+ "\n")
#
# output_file.close()


import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import csv

para = "What can I say about this place. The staff of the restaurant is nice and the eggplant is not bad. Apart from that, very uninspired food, lack of atmosphere and too expensive. I am a staunch vegetarian and was sorely dissapointed with the veggie options on the menu. Will be the last time I visit, I recommend others to avoid"

sentense = word_tokenize(para)
word_features = []

for i,j in nltk.pos_tag(sentense):
    if j in ['JJ', 'JJR', 'JJS', 'RB', 'RBR', 'RBS']:
        word_features.append(i)
# print (word_features)
rating = 0

for i in word_features:
    with open('D:\\Train_sentiment_data.csv', newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            if i == row["sentence"]:
                print (i, row["sentence"])
#                 if row[1] == 'pos':
#                     rating = rating + 1
#                 elif row[1] == 'neg':
#                     rating = rating - 1
# print  (rating)
