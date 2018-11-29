import spacy
import csv
import os
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

NOTE: pass in WITHOUT .CSV, will do this automatically in function
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
        perMillion = float(value) * 1000000 / wordCount(res)  # per million word calculation
        # len function on a spaCy doc returns number of tokens which includes punctuation so perMillion is not accurate
        per_million_count.append((key, value, perMillion))
    # print root_count
    newName += ".csv"
    with open(newName, 'wb') as csvfile: 
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
This section determines if a sentences contains a specified verb
"""
def containsVerb(sent, verb):
    for token in sent:
        if (token.lemma_ == verb):
            return True
    return False

"""
This section returns the root of the sentence.
"""
def getRoot(sent):
    for token in sent:
        if (token.dep_ == u'ROOT'):
            return token
    return

"""
This section gets all the sentences with null objects in a document
"""
def getSentWithNullObject(res):
    hasNull = []
    for sentence in res.sents:
        if (containsNullObject(sentence)):
            hasNull.append(sentence)
    return hasNull

"""
This section gets all the sentences that contain the specified verb
that have null objects within a document
@param verb must be a Unicode string
"""

def getVerbWithNullObject(res, verb):
    hasVerb = []
    allNulls = getSentWithNullObject(res)
    for sentence in allNulls:
        if (containsVerb(sentence, verb)):
            hasVerb.append(sentence)
    return hasVerb

"""
This section returns a dict where the key is the lemma version of a verb
and the value is a list containing all the sentences with that verb.
"""

def sortSentencesByVerb(res):
    sorted = {}
    sentences = getSentWithNullObject(res)
    for sentence in sentences:
        root = getRoot(sentence).lemma_
        if root in sorted:
            sorted[root].append(sentence)
        else:
            sorted[root] = [sentence]
    return sorted



"""
This section gathers a list of all files from the reddit folder and writes a csv file for each one
- No return value, does multiple operations
"""
def parseRedditFiles(res):
    fileList = os.listdir("reddit")
    for file in fileList:
        tempName = file[:-4] # excludes .txt
        # weightlifting.txt -> weightlifting
        countRoots(tempName, res)
    return

"""
Gets the ratio of sentences with null objects to sentences with a specified verb.
"""
def occurance(verb, file):
    doc = parseFile(file)
    countNull = 0
    countObj = 0
    for sentence in doc.sents:
        if (containsVerb(sentence, verb)):
            countObj += 1
            if (containsNullObject(sentence)):
                countNull += 1
    return countNull / float(countObj)

"""
This section finds the difference in occurance of a verb between two TEXT files.
- Uses previous methods so will only search for sentences with null objects.
- Fraction means verb occurs more frequently in file 2, whole number means verb occurs more frequently in file 1.
"""
def occuranceDifference(verb, f1, f2):
    count1 = occurance(verb, f1)
    count2 = occurance(verb, f2)
    if (count1 is 0): 
        return verb + " does not occur in file 1."
    elif (count2 is 0):
        return verb + " does not occur in file 2."
    else: 
        return count1 / count2


"""
ALL CODE BELOW THIS POINT IS FOR COMPARING CSV FILES
"""

"""
This looks through 2 files and finds the list of verbs in both, and returns a list of the shared verbs
"""
def find_shared_verbs(file1, file2):
    verb1 = []
    verb2 = []
    with open(file1) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            verb1.append(row['verb'])

    with open(file2) as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            verb2.append(row['verb'])

    return list(set(verb1).intersection(verb2))


"""
this method will take in two correctly outputted csv files and compare the rates of similar verbs
"""
def compare_csv(file1, file2):
    shared_verbs = find_shared_verbs(file1, file2)
    