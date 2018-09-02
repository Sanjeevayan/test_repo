#Remarks: working code
# from nltk import word_tokenize, pos_tag
# sentence = "At eight o'clock on Thursday film morning word line test best beautiful Ram Aaron design"
# # token = word_tokenize(sentence)
# # pos = pos_tag(token)
# # print (pos)
# nouns = [token for token, pos in pos_tag(word_tokenize(sentence)) if pos.startswith('N')]
# print (nouns)
# #['Thursday', 'film', 'morning', 'word', 'line', 'test', 'Ram', 'Aaron', 'design']
#
# **************************Optoin2:
# Remarks: working code
# import nltk
#
# sentence = "At eight o'clock on Thursday film morning word line test best beautiful Ram Aaron design"
#
# tokens = nltk.word_tokenize(sentence)
#
# tagged = nltk.pos_tag(tokens)
#
# length = len(tagged) - 1
#
# a = list()
#
# for i in range(0, length):
#     log = (tagged [i][1][0] == 'N')
#
#     if log == True:
#         a.append(tagged [i][0])
# print (a)

# import nltk
# from collections import Counter
# # input = "The CBI will submit a chargesheet against Allahabad Bank CEO & MD Usha Ananthasubramanian and several others"
#
# # File = open(fileName) #open file
# # lines = File.read() #read all lines
#
# file = open ("D:\\Test1.txt","r")
# input_file = file.readline()
# output_file = open("D:\\Test4.txt","w")
# sentences = nltk.sent_tokenize(input_file) #tokenize sentences
# nouns = [] #empty to array to hold all nouns
#
# for sentence in sentences:
#      for word,pos in nltk.pos_tag(nltk.word_tokenize(str(sentence))): # understand this sentence/syntax
#          if (pos == 'NN' or pos == 'NNP' or pos == 'NNS' or pos == 'NNPS'):
#                 nouns.append(word)
#                 output_file.write(word + "\n")
# file.close()
# output_file.close()


# # print (nouns)
# print (nouns)

# frequency = Counter (nouns)
# print (frequency)


# from nltk import word_tokenize,pos_tag
# from collections import Counter
# file = open ("D:\\Test2.txt","r")
# input_file = file.read()
# #text = "I am Sanjeev. I stay in Bangalore"
# output_file = open ("D:\\Test4.txt","w")
# tokens = word_tokenize(input_file)
# tags = pos_tag(tokens)
#
# for token,tag in tags:
#     X = token ,"--->", tag
#
#     # print ("Tags of the tokens are",X) # prints all the tokens alonwith corresponding POS tags.
#     output_file.write(X +"\n")

# count = Counter([(token,tag) for token,tag in tags])
# count = Counter([token for token,tag in tags])
# count = Counter([tag for token,tag in tags])

# Trying to filter only a particular tag i.e NNP

# for item in count:
#     if "NNP" in item[1]:
#         print ("NNPs are",item)
#
# print ("Term Frequencies are:",count)

# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import classification_report
# from sklearn import svm
# training_corpus = [
#                    ('I am exhausted of this work.', 'Class_B'),
#                    ("I can't cooperate with this", 'Class_B'),
#                    ('He is my badest enemy!', 'Class_B'),
#                    ('My management is poor.', 'Class_B'),
#                    ('I love this burger.', 'Class_A'),
#                    ('This is an brilliant place!', 'Class_A'),
#                    ('I feel very good about these dates.', 'Class_A'),
#                    ('This is my best work.', 'Class_A'),
#                    ("What an awesome view", 'Class_A'),
#                    ('I do not like this dish', 'Class_B')]
# test_corpus = [
#                 ("I am not feeling well today.", 'Class_B'),
#                 ("I feel brilliant!", 'Class_A'),
#                 ('Gary is a friend of mine.', 'Class_A'),
#                 ("I can't believe I'm doing this.", 'Class_B'),
#                 ('The date was good.', 'Class_A'), ('I do not enjoy my job', 'Class_B')]
# # preparing data for SVM model
# train_data = []
# train_labels = []
# for row in training_corpus:
#     train_data.append(row[0])
#     train_labels.append(row[1])
#
# test_data = []
# test_labels = []
# for row in test_corpus:
#     test_data.append(row[0])
#     test_labels.append(row[1])
#
# # Create feature vectors
# vectorizer = TfidfVectorizer(min_df=4, max_df=0.9)
# # Train the feature vectors
# train_vectors = vectorizer.fit_transform(train_data)
# # Apply model on test data
# test_vectors = vectorizer.transform(test_data)
#
# # Perform classification with SVM, kernel=linear
# model = svm.SVC(kernel='linear')
# model.fit(train_vectors, train_labels)
# prediction = model.predict(test_vectors)
# print(prediction)
# print (classification_report(test_labels, prediction))
#
# validation_text = [ "I love this place"]
#
# new_vector = vectorizer.transform(validation_text)
# new_pred = model.predict(new_vector)
# print (validation_text,"--->",new_pred)
#
# validation_text =  ["I love this place","I hate this hotel"]
#
# for sent in validation_text:
#     new_vector = vectorizer.transform(validation_text)
#     # print (sent,new_vector)
#
#     for vect in new_vector:
#         new_pred = model.predict(vect)
#     print(sent,new_pred )



# for i in range (1,11):
#
#     for j in range (1,12):
#
#         M = j * i
#         print (M,"\t", end = '')
#
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import classification_report
# from sklearn import svm
# import csv
#
#
# def readCSV(fileName):
#     with open(fileName) as File:
#         reader = csv.reader(File, delimiter=',', quotechar=',',
#                             quoting=csv.QUOTE_MINIMAL)
#         return [r[0] for r in reader]
#
#
# def writeCSV(fileName, data):
#     myFile = open(fileName, 'w', newline='')
#     with myFile:
#         writer = csv.writer(myFile)
#         writer.writerows(data)
#
#
# training_corpus = [
#     ('I am exhausted of this work.', 'Class_B'),
#     ("I can't cooperate with this", 'Class_B'),
#     ('He is my badest enemy!', 'Class_B'),
#     ('My management is poor.', 'Class_B'),
#     ('I love this burger.', 'Class_A'),
#     ('This is an brilliant place!', 'Class_A'),
#     ('I feel very good about these dates.', 'Class_A'),
#     ('This is my best work.', 'Class_A'),
#     ("What an awesome view", 'Class_A'),
#     ('I do not like this dish', 'Class_B')]
# test_corpus = [
#     ("I am not feeling well today.", 'Class_B'),
#     ("I feel brilliant!", 'Class_A'),
#     ('Gary is a friend of mine.', 'Class_A'),
#     ("I can't believe I'm doing this.", 'Class_B'),
#     ('The date was good.', 'Class_A'), ('I do not enjoy my job', 'Class_B')]
# # preparing data for SVM model
# train_data = []
# train_labels = []
# for row in training_corpus:
#     train_data.append(row[0])
#     train_labels.append(row[1])
#
# test_data = []
# test_labels = []
# for row in test_corpus:
#     test_data.append(row[0])
#     test_labels.append(row[1])
#
# # Create feature vectors
# vectorizer = TfidfVectorizer()  # min_df=1, max_df=1)
# # Train the feature vectors
# train_vectors = vectorizer.fit_transform(train_data)
# # Apply model on test data
# test_vectors = vectorizer.transform(test_data)
#
# # Perform classification with SVM, kernel=linear
# model = svm.SVC(kernel='linear')
# model.fit(train_vectors, train_labels)
# prediction = model.predict(test_vectors)
# print(prediction)
# print(classification_report(test_labels, prediction))
#
# validation_text = readCSV('D:\\input_data.csv')
#
# new_vector = vectorizer.transform(validation_text)
# # print (sent,new_vector)
#
# new_pred = model.predict(new_vector)
#
# output = []
# for i, sent in enumerate(validation_text):
#     output.append([sent, new_pred[i]])
#
# writeCSV('D:\\output_data.csv', output)
#
# print (output)

#****************************
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics import classification_report
# from sklearn import svm
# import csv
#
# def readCSV(fileName):
#     with open(fileName) as File:
#         reader = csv.reader(File, delimiter=',', quotechar=',',
#                             quoting=csv.QUOTE_MINIMAL)
#         return [r[0] for r in reader]
#
#
# def writeCSV(fileName, data):
#     myFile = open(fileName, 'w', newline='')
#     with myFile:
#         writer = csv.writer(myFile)
#         writer.writerows(data)
#
#
# training_corpus = []
# with open('D:\\Train_sentiment_data.csv', newline='') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#        training_corpus.append(row)
#
#        print(row['sentence'], row['polarity'])
# print ("Training Corpus is: ", training_corpus)
#
# test_corpus = []
# with open('D:\\Test_sentiment_data.csv', newline='') as csvfile:
#     reader = csv.DictReader(csvfile)
#     for row in reader:
#         test_corpus.append(row)
#         # print(row['sentence'], row['polarity'])
#
# print ("Test Corpus is:",test_corpus)
#
# # preparing data for SVM model
# train_data = []
# train_labels = []
# for row in training_corpus:
#     train_data.append(row['sentence'])
#     train_labels.append(row['polarity'])
# print (train_data)
# test_data = []
# test_labels = []
# for row in test_corpus:
#     test_data.append(row['sentence'])
#     test_labels.append(row['polarity'])
# # Create feature vectors
# vectorizer = TfidfVectorizer(min_df=4, max_df=0.9)
# #vectorizer = TfidfVectorizer() #(min_df=1, max_df=1) # Arun's suggestion
# # print (vectorizer)
# # Train the feature vectors
# train_vectors = vectorizer.fit_transform(train_data)
#
# # Apply model on test data
# test_vectors = vectorizer.transform(test_data)
#
# # Perform classification with SVM, kernel=linear
# model = svm.SVC(kernel='linear')
# model.fit(train_vectors, train_labels)
# prediction = model.predict(test_vectors)
# print(prediction)
# print (classification_report(test_labels, prediction))
#
# #Validation on new Data set
#
# validation_text = readCSV('D:\\input_data.csv')
#
# new_vector = vectorizer.transform(validation_text)
# # print (sent,new_vector)
#
# new_pred = model.predict(new_vector)
#
# output = []
# for i, sent in enumerate(validation_text):
#     output.append([sent, new_pred[i]])
#
# writeCSV('D:\\output_data.csv', output)
#
# print (output)

# from nltk import sent_tokenize,word_tokenize
# key_words = ["good","excellent", "well","great"]
# # negative_words = ["bad", "terrible", "shock"]
# input_text = "He is a good man. He plays well. He is bad. He is playing. She is good. "
# sents = sent_tokenize(input_text)
# # tokens = word_tokenize(input_text)
#
# for sentence in sents:
#     sent_tokens = word_tokenize(sentence)
#     for token in sent_tokens:
#         if token in key_words:
#             print (sentence)

#             print(num)




#
# num = 16
# K = []
# for i in range (1, num+1):
#     if num % i == 0:
#         K.append (i)
# smallest = K[0]
# print (smallest)
# for i in K:
#     if i < smallest:
#         smallest = i
# print (smallest)

# alist=[-45,0,3,10,90,5,-2,4,18,45,100,1,-266,706]
#
# largest = alist[0]
# for item in alist:
#     if item > largest:
#        largest = item
# print (largest)

# h_letters = []
#
# for letter in "human":
#     h_letters.append(letter)
# print (h_letters)

# h_letters = [letter for letter in 'human']
# print (h_letters)

# num_list = [y for y in range(100) if y % 2 == 0 if y % 5 == 0]
# print(num_list)

# num_list = [num for num in range (100)if num % 4== 0 if num % 6 ==0]
# print (num_list)

# a = [2,4,6,8,10]
# b = [i*10 for i in a]
# print (b)
# import nltk
# from nltk import sent_tokenize,word_tokenize
# a = "I love India. I stay in Delhi. I like India. I play ball . I hate him."
# key_word = ["love", "stay" , "play"]
# synonym_list = ["like"]
# sent = sent_tokenize(a)
# K = []
# for line in sent:
#     tokens = word_tokenize(line)
#     # print (tokens)
#     for word in tokens:
#         if word in key_word:
#             # if word in synonym_list:
#             K.append (line)
#         if word in synonym_list:
#             K.insert (-1,line)
#             # print (line)
#
# print (K)

# import nltk
# from nltk import sent_tokenize,word_tokenize,pos_tag
# a = "I love India. I stay in Delhi. I like India. I play ball . I hate him."
# sents = sent_tokenize(a)
# for sentence in sents:
#     tokens = word_tokenize(sentence)
#     pos = pos_tag(tokens)
#
#     for word,tag in pos:
#         print (word, "--->",tag)


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
                print("Input Sentence is:" , sentence , "Polarity --> positive ")
            if tag[0]in negative_words:

                print("Input Sentence is:" ,sentence, "Polarity --> negative ")
                neg_sentiment.append(sentence)
        else:
            print (tag[1],"It doesnot have sentiment")

print ("Positive sentences are:",pos_sentiment)

print ("Negative sentences are:",neg_sentiment)
# print (len(pos_sentiment))
# print (len(neg_sentiment))
# if len(pos_sentiment) > len(neg_sentiment):
#     print ("Text is positive ")
# else :
#     print ("Text is Negative")






