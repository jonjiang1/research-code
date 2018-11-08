import spacy
import csv
import os
from spacy.lang.en.examples import sentences
from collections import defaultdict
import math
import nltk
from nltk.corpus import wordnet as wn
import operator
from nltk.tokenize import sent_tokenize
import pickle
import dill
import re
nlp = spacy.load('en_core_web_sm')


"""

The goal of this file is to.....
1. read in a csv of transitive verbs as well as (one or more) text files of English data.
2. output a new csv which provides, for each transitive verb....
.... its pmw (per million) occurrence as a VERB in the text files of English data
........ its "simple" selectional strength, meaning the % of the time it occurs with its most common object
......... the word that is actually its most common object
...... the number of times this verb appears overall
..... the % of sentences in which it has an OVERT OBJECT
...... the % of sentences in which  it does NOT HAVE an overt object
...... a list of the sentences where it does NOT HAVE an overt object.

"""


"""
Code taken from Stack Overflow to make a dict with mixed types of data.

"""
class KeyBasedDefaultDict(dict):
    def __init__(self, default_factories, *args, **kw):
        self._default_factories = default_factories
        super(KeyBasedDefaultDict, self).__init__(*args, **kw)

    def __missing__(self, key):
        factory = self._default_factories.get(key)
        if factory is None:
            raise KeyError(key)
        new_value = factory()
        self[key] = new_value
        return new_value

"""
Code that takes a list of dicts and writes it to a csv.
"""

# takes a list of dicts (all with same keys) and returns a list of the keys
def get_list_of_keys(list_of_dicts):
    set_of_keys = set()
    for d in list_of_dicts:
        keys = d.keys()
        for key in keys:
            set_of_keys.add(key)
    return list(set_of_keys)



##writes a list of dicts to a csv file
def dicts_to_csv(list_of_dicts, filename):
    f = open(filename, 'wb')
    list_of_keys = get_list_of_keys(list_of_dicts)
    dict_writer = csv.DictWriter(f, list_of_keys)
    dict_writer.writer.writerow(list_of_keys)
    dict_writer.writerows(list_of_dicts)



"""
Reads in a textfile, and tries to remove all the messy stuff
like non-ASCII characters and links to websites.
You run this BEFORE you run "cull relevant sentences".
"""
def remove_nonascii(raw_txtfile, clean_txtfile):
    reader = open(raw_txtfile, "r")
    writer = open(clean_txtfile, "w")
    for line in reader: # we only want to keep/write ASCII characters.
        linewords = ""
        for word in line.split(" "):
            if "http" not in word and "deleted" not in word: # trying to remove websites, deleted stuff, etc.
                clean_word = "".join(i for i in word if ord(i) < 128)
                linewords += " " + clean_word.replace("\\n", " ").replace("&gt;","").replace("&amp;","")      
        writer.write(linewords)
    writer.close()
        


"""
Reads in a (cleaned-up, ASCII only) text file.
Writes a new file that ONLY contains sentences including the transitive verbs (as lemmas) that we're interested in,
after checking the lemmas in each sentence against a set of verbs to keep, taken from our csv of transitive verbs.
(This strategy is meant to significantly shorten the file, since the file only contains sentences with our verbs in it.).
The output of this function is the file you use to run txts_csv_to_csv.

"""


def cull_relevant_sentences(clean_txtfile, culled_txtfile, csv_of_verbs):
    lemmer = nltk.WordNetLemmatizer()
    sent_tokenizer = nltk.sent_tokenize
    csvreader = csv.DictReader(open(csv_of_verbs, "r"))
    rel_sents = 0 # keeping track of number of relevant/kept sentences
    skipped_sents = 0 # just keeping track of number of skipped sentences, to know how much we're shortening the file
    verbs = set()
    for line in csvreader:
        verb = line["verb"]
        verbs.add(verb)
    txtreader = open(clean_txtfile, "r")
    writer = open(culled_txtfile, "w")
    lastwritten = ""
    for line in txtreader:
        if "automoderator" not in line and "**PLEASE NOTE:**" not in line and "bot" not in line and "mods" not in line:
            # trying to remove lines that are written by the automoderator and bots and stuff.
            if not lastwritten == "\n":
                writer.write("\n")
                lastwritten = "\n" # this is weird, sorry, just want to only write one newline in a row.
            for sentence in sent_tokenizer(line):
                sentwords = ""
                relevant_verb_in_sent = False # this is False unless it gets set to True
                for word in sentence.split(" "):
                  sentwords += " " + word
                  if lemmer.lemmatize(word) in verbs:
                        relevant_verb_in_sent = True # if we see relevant verb in the sentence,
                                                            # we keep the sentence.
                if relevant_verb_in_sent:
                    rel_sents += 1
                    print "relevant sentences so far " + str(rel_sents)
                    writer.write(sentwords)
                    lastwritten = sentwords
                else:
                    skipped_sents += 1
                    print "skipping " + str(skipped_sents) + " sentences"
                    # skipping all sentences that don't have any of the relevant verbs in them.
    print "total of " + str(rel_sents) + " relevant sentences"
    writer.close()
        

"""
This function reads in a text file and yields the Spacy-parsed sentences in it.
"""
def getParsedSents(textFile):
    text = open(textFile, 'r').read()
    uText = text.decode('utf-8')
    doc = nlp(uText)
    return doc.sents





"""
gets the VERBS in an already-parsed sentence
"""
def get_verbs(sent):
    verbs = []
    for token in sent:
        if str(token.pos_) == "VERB":
            verbs.append(str(token.lemma_))
    return verbs


"""
Takes a sentence and a verb.  Returns one of 3 things....

1. Returns the object of that verb if there is one (according to dependency parse).
2. If the sentence is in the passive or doesn't have a NOUN as its obj/dobj,
     returns "ineligible sentence".
3.  If there is NO OBJECT but the sentence is eligible (not passive), returns "No object".
"""
def get_object(sent, verb):
    is_object = False
    ineligible = False
    obj = ""
    for token in sent:
        if (token.dep_ == u"auxpass") or (token.dep_ == "nsubjpass"): # eliminating any passive sentences.
            ineligible = True
        elif (token.dep_ == u"obj" or token.dep_ == u"dobj") and (str(token.head.lemma_) == verb):
                # makes sure this token is an OBJECT in the parse
                # and also that its "HEAD" in the dependency parse is the specified verb.
                is_object = True
                obj = token.lemma_
    if ineligible:
        return "Ineligible" # passive or non-alphabetic object gets excluded.
    elif is_object == True:
        print verb + "_" + str(obj)
        return str(obj)
    elif is_object == False and ineligible == False: # eligible sentences with no object.
        return "No object"

        




"""
This takes in a sentence and prints out its dependency parse.
Useful for checking how Spacy handles various sentences.
Not actually called anywhere, just for illustration purposes.
"""
def parse_sentence(sentence):
    uText = sentence.decode('utf-8')
    doc = nlp(uText)
    for token in doc:
        print str(token) + "(" + str(token.dep_) + "; " + str(token.pos_) + ")"






"""
takes in the count of a word in some text, and the total word count of that text,
then returns the per-million-word count of that word in that text, as an integer.
"""
def get_pmw(count_of_word, total_count):
    return int((count_of_word / total_count) * 1000000)






"""
Now let's turn to the "simpler selectional strength" idea.
from Dan Jurafsky's course notes
here: https://web.stanford.edu/~jurafsky/slp3/slides/22_select.pdf, slide 37
Using this simpler method, you just get the probability of a noun n occurs as the object of a verb v.
(you don't use any taxonomic classes from WordNet)
This takes a lot less time than the Resnik version.

(Count of occurrences of n as object of v) / (count of occurrences of v with any object).

So if "eat" occurs 3 times in our data, "eat pizza" and "eat cake" and "eat pizza", then
its most common object is "pizza" and it occurs 2/3 of the time.
"""

"""
This first function creates a dict of all verbs, and then for each verb, the number of times it appears with each
of its different objects.
"""
def get_verb_obj_dict(verb_obj_pairs):
    verb_obj_dict = defaultdict(lambda: defaultdict(int))
    for pair in verb_obj_pairs:
        verb = pair.split(" ")[0]
        obj = pair.split(" ")[1]
        verb_obj_dict[verb][obj] += 1
    return verb_obj_dict
   
#print get_verb_obj_dict(verb_obj_pairs)

"""
this tells you how often a given object appears as a percentage of all objects of a given verb.
"""
def how_often_obj_with_verb(verb, obj, verb_obj_dict):
    total_v = 0
    total_v_with_obj = 0
    for objkey in verb_obj_dict[verb].keys():
        total_v += verb_obj_dict[verb][objkey]
    return round(float(verb_obj_dict[verb][obj]) / total_v, 3)


"""
We'd also like, for each verb, its
"Simple" selectional STRENGTH:
What percent of its occurrences (with an object) involve its MOST-COMMON object?
Also, what are its most common objects, and how often do they appear?

"""

def get_simple_sel_strength(verb, verb_obj_dict):
    max_obj = max(verb_obj_dict[verb].iteritems(), key=operator.itemgetter(1))[0]
    objs = ""
    for obj, count in reversed(sorted(verb_obj_dict[verb].iteritems(), key=lambda (obj,count): (count,obj))):
            objs += obj + "(" + str(count) + "),"
    return (str(how_often_obj_with_verb(verb, max_obj, verb_obj_dict)),
            objs)



"""
ok, finally: we're gonna read in all our verb-object pairs and then make a dict of dicts.
for each verb, we have a dict which tells us....
.... its "simple selectional strength": the percent of the time it occurs with its MOST COMMON object
..... and (just so we know) its most common objects in decreasing order.
"""
def get_simple_sel_dict(verb_obj_pairs):
    simple_sel_dict = defaultdict(lambda: defaultdict(str))
    verb_obj_dict = get_verb_obj_dict(verb_obj_pairs)
    for verb in verb_obj_dict.keys():
        simple_sel_dict[verb]["sel"] =get_simple_sel_strength(verb, verb_obj_dict)[0]
        simple_sel_dict[verb]["maxobjs"] = get_simple_sel_strength(verb, verb_obj_dict)[1]
    return simple_sel_dict



"""
that's the END of the "simple" selectional strength stuff.
"""



"""
takes in a csv of transitive verbs and returns a dict of dicts with all those verbs as the keys.
using some code taken from Stack Overflow to make a default dict that allows mixed data types (float and list).
Just setting up the type of the "values" so they can be stored in the correct type below.
"""
def get_verbdict(input_csv):
    mapping = {'overall_occurrences': float,
               'number_with_obj': float,
               "number_withOUT_obj": float,
               "objectless_sentences": list,
               "sents_WITH_obj": list
               }
    verbdict = defaultdict(lambda: KeyBasedDefaultDict(mapping))
    reader = csv.DictReader(open(input_csv, "r"))
    for line in reader:
        verb = line["verb"]
        verbdict[verb]["Levin_classes"] = line["Levin_classes"]
    return verbdict
        






"""
This function reads in a text file of English data (e.g Reddit)
and csv of transitive verbs that we're interested in.

Then it outputs a new csv with all the desired information in it....

.... its pmw (per million) occurrence as a VERB in the text files of English data (Reddit, for example)
........ its "simple" selectional strength, meaning the % of the time it occurs with its most common object
......... the word that is actually its most common object
...... the number of times this verb appears overall
..... the % of sentences in which  it has an OVERT OBJECT
...... the % of sentences in which it does NOT HAVE an overt object
...... a list of the sentences where it does NOT HAVE an overt object.
........ a list of the objects that occur with it when it DOES HAVE an object.

This is the function that does the most work.

"""
def txt_csv_to_csv(txtfile, input_csv, output_csv):
    verbdict = get_verbdict(input_csv)
    sents = getParsedSents(txtfile)
    verb_obj_pairs = []
    total_word_count = 0
    counter = 0
    for sent in sents:
        counter += 1
        print "processing " + str(counter) + " sentences...."
        ## first, we add all the words in the sentence to our total word count....
        for token in sent:
            if (token.pos_ != u'PUNCT' or token.pos_ != u'SYM'):
                total_word_count += 1
        # next, we find the root of the sentence (if it's in our csv of transitive verbs)
                # and then check whether it has an object....
        verbs = get_verbs(sent)
        for verb in verbs:
            if verb in verbdict.keys():
                obj = get_object(sent,verb)
                print "SENTENCE:"
                print sent
                print "OBJECT:"
                print obj
                # if the verb is passive or whatever, it's "ineliglbe".... 
                if obj == "Ineligible":
                    verbdict[verb]["overall_occurrences"] += 1
                # if there is no object, we add the sentence to our list of "objectless sentences"...
                elif obj == "No object":
                    verbdict[verb]["overall_occurrences"] += 1
                    verbdict[verb]["number_withOUT_obj"] += 1
                    verbdict[verb]["objectless_sentences"].append(sent)
                # if there is an object, we add it to our list of verb-object pairs.....
                else:
                    verbdict[verb]["overall_occurrences"] += 1
                    verbdict[verb]["number_with_obj"] += 1
                    verbdict[verb]["sents_WITH_obj"].append(sent)
                    verb_obj_pairs.append(str(verb) + " " + str(obj))
           
    print "creating simple selection dict....."
    simple_sel_dict = get_simple_sel_dict(verb_obj_pairs)
    
    # COMMENT THIS OUT IF YOU WANT IT TO RUN FAST ENOUGH.
    #print "creating resnik dict.."
    #resnik_dict = get_resnik_sel_dict(verb_obj_pairs)
    
    # now we write all of this to our new csv file.
    list_of_dicts = [] # ultimately written to new csv....
    # i realize it's not good that this is so repetitive with what comes above but
    # the only way I write csv files is with a list of dicts so..... haha
    print "writing output....."
    attested_verbs = (verb for verb in verbdict.keys() if (verbdict[verb]["number_with_obj"] + verbdict[verb]["number_withOUT_obj"]) > 0)
    for verb in attested_verbs:
        d = defaultdict(str)
        d["verb"] = verb
        d["Levin_classes"] = verbdict[verb]["Levin_classes"]
        d["overall_occurrences"] = verbdict[verb]["overall_occurrences"]
        d["number_with_obj"] = verbdict[verb]["number_with_obj"]
        d["number_withOUT_obj"] = verbdict[verb]["number_withOUT_obj"]
        d["percent_implicit_obj"] = verbdict[verb]["number_withOUT_obj"] / (verbdict[verb]["number_withOUT_obj"] + verbdict[verb]["number_with_obj"])
              
       # COMMENT THIS OUT IF YOU WANT IT TO RUN FASTER!
       
        d["simple_sel"] = simple_sel_dict[verb]["sel"]
        d["most_common_objs"] = simple_sel_dict[verb]["maxobjs"]
        
        d["pmw_count"] = get_pmw(verbdict[verb]["overall_occurrences"], total_word_count)
        d["objectless_sentences"] = verbdict[verb]["objectless_sentences"]
        d["sents_WITH_obj"] = verbdict[verb]["sents_WITH_obj"]
        list_of_dicts.append(d)
    dicts_to_csv(list_of_dicts, output_csv)




if __name__ == '__main__':
    txt_csv_to_csv("toy.txt", "toy_csv.csv", "toy_output.csv")
    #this works!! (if you want to edit stuff, do it with "toy.txt" and "toy.csv")
  
    #remove_nonascii("askreddit.txt", "clean_askreddit.txt")
    #cull_relevant_sentences("clean_askreddit.txt", "culled_askreddit.txt", "deduped_trans_verbs.csv")
    #print parse_sentence("They drank water.")
    txt_csv_to_csv("culled_askreddit.txt", "deduped_trans_verbs.csv", "askreddit_output.csv")