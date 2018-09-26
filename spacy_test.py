import spacy
from spacy.lang.en.examples import sentences

nlp = spacy.load('en_core_web_sm')
text = open('weightlifting.txt', 'r').read()
uText = text.decode('utf-8')
doc = nlp(uText)
# print(doc)
dependency_count = {}
for token in doc:
    if (token.dep_ not in dependency_count):
        dependency_count[token.dep_] = 1
    else:
        dependency_count[token.dep_] = dependency_count[token.dep_] + 1

    # print(token.text, token.pos_, token.dep_)
print dependency_count