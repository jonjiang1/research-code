import spacy
import csv
from spacy.lang.en.examples import sentences

nlp = spacy.load('en_core_web_sm')


# tempFile = 'weightlifting.txt'
def parseFile(textFile):
    text = open(textFile, 'r').read()
    uText = text.decode('utf-8')
    doc = nlp(uText)
    return doc
# result = parseFile(tempFile)

"""
This section counts the total number of words in the document.
It does not include punctuation or symbols such as math symbols or
emoticons.
"""

def wordCount(res):
    count = 0
    for token in res:
        if (token.pos_ != u'PUNCT' or token.pos_ != u'SYM'):
            count = count + 1
    return count

"""
This section counts the total number of different dependency classes in a document,
stores it in a dict, and then prints them out with corresponding counts
@param res refers to the file to be parsed (short for result)
"""

def getDependencyCount(res):
    dependency_count = {}
    for token in res:
        if (token.dep_ not in dependency_count):
            dependency_count[token.dep_] = 1
        else:
            dependency_count[token.dep_] = dependency_count[token.dep_] + 1

    return dependency_count

# getDependencyCount()

"""
This section counts the number of roots in a text file,
calculates the per million word count, and then writes them to a CSV file.
@param newName is the new name of the csv file we are writing
@param res is the same result are using in .getDependencyCount
automatically writes the file, doesn't return anything
"""

def countRoots(newName, res):
    root_count = {}
    for token in res:
        if(token.dep_ == u'ROOT'):
            if (token.lemma_.encode('utf-8') not in root_count):
                root_count[token.lemma_.encode('utf-8')] = 1
            else:
                root_count[token.lemma_.encode('utf-8')] = root_count[token.lemma_.encode('utf-8')] + 1
    per_million_count = []
    for key, value in root_count.iteritems():
        perMillion = float(value) * 1000000 // wordCount(res)  # per million word calculation
        # len function on a spaCy doc returns number of tokens which includes punctuation so perMillion is not accurate
        per_million_count.append((key, value, perMillion))
    # print root_count
    with open(newName, 'w') as csvfile: 
        writer = csv.writer(csvfile)
        # TODO - csv has newlines in-between?
        writer.writerow(['verb', 'count', 'per_million_words']) # Writes header for CSV
        for key, value, million in per_million_count:
            writer.writerow([key, value, million])
    return 

# countRoots(temp)

"""
This section determines if a sentence contains a verb with a null object
- Returns True if NO object detected
"""
def containsNullObject(sent):
    for token in sent:
        if (token.dep_ == u'OBJ' or token.dep_ == u'DOBJ'):
            return False
    return True

"""
This section gets all the sentences with null objects in a document
"""
def getSentWithNullObject(res):
    hasNull = []
    for sentence in res.sents:
        if (containsNullObject(sentence)):
            hasNull.append(sentence)
    return hasNull

