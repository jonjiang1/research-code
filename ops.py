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

### This section counts the total number of different dependency classes in a document,
### stores it in a dict, and then prints them out with corresponding counts
### @param res refers to the file to be parsed (short for result)
def getDependencyCount(res):
    dependency_count = {}
    for token in res:
        if (token.dep_ not in dependency_count):
            dependency_count[token.dep_] = 1
        else:
            dependency_count[token.dep_] = dependency_count[token.dep_] + 1

    return dependency_count

# getDependencyCount()

### This section counts the number of roots in a text file, stores them in a dict,
### and then writes them to a CSV file

### temp = 'weightlifting.csv'
### @param newName is the new name of the csv file we are writing
### @param res is the same result are using in .getDependencyCount
### automaticallyw rites the file, doesn't return anything
def countVerbs(newName, res):
    root_count = {}
    for token in res:
        if(token.dep_ == u'ROOT'):
            if (token.lemma_.encode('utf-8') not in root_count):
                root_count[token.lemma_.encode('utf-8')] = 1
            else:
                root_count[token.lemma_.encode('utf-8')] = root_count[token.lemma_.encode('utf-8')] + 1
    # print root_count
    with open(newName, 'w') as csvfile:
        fieldnames = ['verb', 'count']
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for key, value in root_count.iteritems():
            writer.writerow({'verb': key, 'count': value})
    return 

# countVerbs(temp)