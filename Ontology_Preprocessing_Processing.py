from PyPDF2 import PdfFileReader
import re
import nltk
from nltk import sent_tokenize, ne_chunk, word_tokenize
from nltk.tag import pos_tag, pos_tag_sents
from nltk.tree import Tree

document = 'd:\\ICISA_DE.pdf'
#Pre-processing1: Converting PDF to Text

def extractText(path):
    text = ''
    with open(path, 'rb') as f:
        pdf = PdfFileReader(f)
        num_pages = pdf.getNumPages();

        for pg in range(num_pages):
            page = pdf.getPage(pg)
            text += page.extractText()
        return text

#Pre-processing2:Cleaning the input text
def cleanupText(text):
    text = re.sub('(\.\ ?){2,}', ' ', text)
    text = re.sub('\n', ' ', text)
    text = re.sub('\ {2,}', ' ', text)

    return text

#NLP Based Text processing1:Tokenization and Parts of speech Tagger
def tagText(text):
    sents = [word_tokenize(s) for s in sent_tokenize(text)]

    taggedSents = pos_tag_sents(sents)

    return taggedSents

#NLP Based Text processing2:NP Chunking for identification of concepts

def neChunk(sents):
    newSents = []
    for sent in sents:
        newSent = []
        for i in ne_chunk(sent):
            if (type(i) == Tree):
                entity_name = ' '.join([c[0] for c in i.leaves()])
            else:
                entity_name = i[0]
            newSent.append(entity_name)

    newSents.append(newSent)


def chunk(sentence):
    chunkToExtract = """
    NP: {<NNP>*}
		{<DT>?<JJ>?<NNS>}
		{<NN><NN>}"""
    parser = nltk.RegexpParser(chunkToExtract)
    result = parser.parse(sentence)
    sent = []
    for subtree in result.subtrees():
        if subtree.label() == 'NP':
            t = subtree
            t = ' '.join(word for word, pos in t.leaves())
            sent.append(t)

    return sent
# Creating a vocabulary of concepts for creating an ontology

def extractVocab(taggedsents):
    dict = {word: 0 for sent in taggedSents for word in chunk(sent)}

    vocabulary = list(dict.keys())

    return vocabulary


text = extractText(document)
cleanedText = cleanupText(text)
taggedSents = tagText(cleanedText)

vocabulary = extractVocab(taggedSents)

print (vocabulary)