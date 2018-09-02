# Toy Summarization Code. Remarks: Working
#Step1:
from nltk import sent_tokenize,word_tokenize,ne_chunk
a = "I love India. I stay in Delhi. I like India. I play ball . I hate him."
key_word = ["love", "stay" , "play"]
synonym_list = ["like"]
sent = sent_tokenize(a)
K = []
for line in sent:
    tokens = word_tokenize(line)
    # print (tokens)
    for word in tokens:
        if word in key_word :
            # if word in synonym_list:
            K.append(line)
        elif word in synonym_list:
            # print (line)
            K.append (line)
print (K)

# step2:
import nltk
from nltk import word_tokenize,sent_tokenize
doc = "I love food." \
      " I stay in Delhi." \
      " I like India. " \
      "I play ball ." \
      " I hate him." \
      " He works at  Baidu."
tokenized_doc = nltk.word_tokenize(doc)
print (tokenized_doc)

key_word = ["love","country", "play"]
synonym_list = ["like"]
sent = sent_tokenize(doc)

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
L = []
for i in named_entities:
    L.append(i[0])
print (L)
K = []
for line in sent:
    tokens = word_tokenize(line)
    # print (tokens)
    for word in tokens:
        if word in key_word :
            K.append(line)
            # print(line)
        elif word in synonym_list:
            # print (line)
            K.append (line)
        elif word in L:
            K.append(line)


print ("summary is",K)

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
# print (K)


# from nltk import sent_tokenize,word_tokenize,pos_tag
#
# sent_dict = ["good","excellent", "well","great"]
# # positive_words = ["good","excellent", "well","great"]
# # negative_words = ["bad", "terrible", "shock"]
# # sentiment_tag = ["JJ","RB"]
#
# input_text = "He is a bad man . He plays well. He is sleeping. "
# sents = sent_tokenize(input_text)
# # print (sents)
# sent_sentence = []
#
# for sentence in sents:
#     sent_tokens = word_tokenize(sentence)
#
#     for token in sent_tokens:
#         if token in sent_dict:
#             print (sentence,"It has Sentiment")
#
#         else:
#             print (sentence,"It doesnot have sentiment")


# from nltk import sent_tokenize,word_tokenize
#
# sent_dict = ["good","excellent", "well","great"]
#
#
# input_text = "He is a bad man . He plays well. He is sleeping. "
# sents = sent_tokenize(input_text)
# # print (sents)
# sent_sentence = []
#
# for sentence in sents:
#     sent_tokens = word_tokenize(sentence)
#
#     for token in sent_tokens:
#         if token in sent_dict:
#             print (sentence, "--->It has sentiment")
#         else:
#             print (sentence, "--->It doesnot have sentiment")

# from nltk import sent_tokenize,word_tokenize,pos_tag
# input_text = "He is a bad man . He plays well. He is sleeping."
# keyword_dictionary = ["bad", "well"]
# sents = sent_tokenize(input_text)
# for sentence in sents:
#     tokens = word_tokenize(sentence)
#     for token in tokens:
#         if token in keyword_dictionary:
#             print (token , "--->", sentence)

# *********************************************

#print('!* To Find Prime Number')
# def prime_number():
#     flag = 1
#     n = int(input('Enter the number'))
#     for i in range(2,int(n)):
#         if(n%i == 0):
#             print('%d is not a prime number' %n)
#             flag = 0
#             break
#     if(flag == 1):
#         print('%d is a prime number' %n)
# print (prime_number ())

# Program to print prime number using a flag
# number = int (input("Enter a number"))
# flag = 1
# for i in range (2,int (number/2)):
#     if number%i ==0:
#         print ("It is not a prime number")
#         flag = 0
#         break
# if (flag == 1):
#     print ("It is a prime number")
#
# # Program to print Even number using a flag
# number = int (input ("Enter a number"))
# flag =1
# if number % 2 == 0:
#     print ("Even Number")
#     flag = 0
#     if (flag ==1):
#         print ("Even NUmber")
# else:
#     print ("Odd NUmber")
#
# number = int (input ("Enter a number"))
# flag =0
# if number % 2 == 0:
#     print ("Even Number")
#     flag = 1
#     if (flag ==0):
#         print ("Even NUmber")
# else:
#     print ("Odd NUmber")


from nltk import sent_tokenize, word_tokenize

sent_dict = ["good", "excellent","bad", "well", "great"]

input_text = "He is a bad man . He plays well. He is sleeping. "
sents = sent_tokenize(input_text)

sent_sentence = []


for sentence in sents:
    sent_tokens = word_tokenize(sentence)

    Flag = False

    for token in sent_tokens:
        if token in sent_dict:
            Flag = True

    if (Flag == True):
        print(sentence, "--->It has sentiment")
    else:
        print(sentence, "--->It does not have sentiment")
