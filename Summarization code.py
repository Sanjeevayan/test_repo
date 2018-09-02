
#https://gist.github.com/Sebastian-Nielsen/3bc45cbba6cb25837f5a6f11dbeeb044
#https://dev.to/davidisrawi/build-a-quick-summarizer-with-python-and-nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.stem.snowball import SnowballStemmer
import nltk



text = "India  also called the Republic of India  is a country in South Asia. " \
       "It is the seventh-largest country by area, the second-most populous country (with over 1.2 billion people), and the most populous democracy in the world." \
       "It is bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast. " \
       "It shares land borders with Pakistan to the west; China, Nepal, and Bhutan to the northeast; and Bangladesh and Myanmar to the east." \
       " In the Indian Ocean, India is in the vicinity of Sri Lanka and the Maldives. India's Andaman and Nicobar Islands share a maritime border with Thailand and Indonesia."\
        "The Indian subcontinent was home to the urban Indus Valley Civilisation of the 3rd millennium BCE."\
        "In the following millennium, the oldest scriptures associated with Hinduism began to be composed. " \
        "Social stratification, based on caste, emerged in the first millennium BCE, and Buddhism and Jainism arose." \
    " Early political consolidations took place under the Maurya and Gupta empires; the later peninsular Middle Kingdoms influenced cultures as far as southeast Asia." \
" In the medieval era, Judaism, Zoroastrianism, Christianity, and Islam arrived, and Sikhism emerged, all adding to the region's diverse culture. " \
"Much of the north fell to the Delhi sultanate; the south was united under the Vijayanagara Empire." \
" The economy expanded in the 17th century in the Mughal Empire. In the mid-18th century, the subcontinent came under British East India Company rule, and in the mid-19th under British crown rule. " \
"A nationalist movement emerged in the late 19th century, which later, under Mahatma Gandhi, was noted for nonviolent resistance and led to India's independence in 1947."\


stemmer = SnowballStemmer("english")
stopWords = set(stopwords.words("english"))
words = word_tokenize(text)

freqTable = dict()
for word in words:
    word = word.lower()
    if word in stopWords:
        continue #doubt
    # word = stemmer.stem(word)
    if word in freqTable:
        freqTable[word] += 1 #doubt
    else:
        freqTable[word] = 1 #doubt
# print (freqTable)
sentences = sent_tokenize(text)
sentenceValue = dict()

for sentence in sentences:
    for word, freq in freqTable.items():
        if word in sentence:
            if sentence in sentenceValue:
                sentenceValue[sentence] += freq #doubt
            else:
                sentenceValue[sentence] = freq #doubt
                print (sentence)

print ("Sentence Value is",sentenceValue)
# print (len(sentenceValue))

sumValues = 0
for sentence in sentenceValue:
    sumValues += sentenceValue[sentence] #doubt
    print (sentence,sumValues)
# Average value of a sentence from original text
average = int(sumValues / len(sentenceValue))

summary = ''
for sentence in sentences:
    if (sentence in sentenceValue) and (sentenceValue[sentence] > (1.2 * average)):
        summary += " " + sentence #doubt
        # summary += " \n" + sentence  # doubt

# print (summary)