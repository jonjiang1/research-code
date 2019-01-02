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
......... its selectional strength calculated by the Resnik formulation
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
    
    # swaps verb to first index
    first = list_of_keys[0]
    verb_index = list_of_keys.index("verb")
    if first != verb_index:
        list_of_keys[verb_index] = first
        list_of_keys[0] = "verb"


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





""""
SELECTIONAL STRENGTH.....
this is the code I wrote (from the Jupyter file) that takes in pairs of verbs and objects (from corpus data)
and calculates the selectional strength from the verb.
It's kind of messy/annoying because things get really big really fast
Makes a lot of use of "pickle" - a library for storing stuff instead of recomputing it every time you need it.
"""""
# this just prints a dict by its values.
def print_dict_by_values(d):
    sorted_d = [ (v,k) for k,v in d.iteritems() ]
    sorted_d.sort(reverse=True) # natively sort tuples by first element
    for v,k in sorted_d:
        print "%s: %d" % (k,v)

"""
Forst, we collect data on objects and the class they fall into, classes and the objects that fall into them,
and the frequencies of both objects and verb-object pairings.

We read in the verb-object pairs (e.g., "eat pizza") and create several different dictionaries.

class_dict: for each taxonomic class in WordNet,  the set of words that fall into it.

word_dict: for each word, the set of taxonomic WordNet classes that it's a member of.

word_freq_dict: for each word (object of a verb), its raw frequency

verb_obj_dict: for each verb-object pair, its raw frequency

We make each of these dictionarites ONCE and then "pickle" it, for efficiency.

"""

def get_class_and_word_dicts(verb_obj_pairs):
    class_dict = defaultdict(set) # for each class, its words. a set so we don't double-count.
    word_dict = defaultdict(set) # for each word, its classes.  a set so we don't double-count.
    word_freq_dict = defaultdict(int) # for each word (object).
    verb_obj_dict = defaultdict(lambda: defaultdict(int)) # for each verb, the objs it has and their frequency
    counter = 0
    for pair in verb_obj_pairs:
        counter += 1
        print "creating class and word dicts..... " + str(counter)
        verb = pair.split(" ")[0]
        obj = pair.split(" ")[1]
        word_freq_dict[obj] += 1
        verb_obj_dict[verb][obj] += 1
        lemmas = wn.lemmas(obj)
        for lemma in lemmas:
            synset = lemma.synset()
            if ".n" in synset.name(): # only want synsets of NOUNS for our purposes.
                hypers = synset.hypernyms()
                for hyper in hypers:
                    hyper_name = hyper.name()
                    paths = hyper.hypernym_paths()
                    for path in paths:
                        for item in path:
                            item_name = item.name()
                            class_dict[item_name].add(obj)
                            word_dict[obj].add(item_name)    
    # now we "pickle" this, an efficient way of storing it so we don't have to recreat it each time.
    pickle.dump(class_dict, open("class_dict.p", "wb"))
    pickle.dump(word_dict, open("word_dict.p", "wb"))
    pickle.dump(word_freq_dict, open("word_freq_dict.p", "wb"))
    pickle.dump(verb_obj_dict, open("verb_obj_dict.p", "wb"))
                                    


"""
Frequency of a class c
Next we get the "frequency" of each taxonomic class,
as described by Resnik 1993 (Dissertation) p. 28 equation 2.12
"""
# to eventually get probability of a class, we first, get frequency of a class.
def get_class_freq(c):
    class_dict = pickle.load(open("class_dict.p", "rb")) # for each class, its words
    word_dict = pickle.load(open("word_dict.p", "rb")) # for each word, its classes
    word_freq_dict = pickle.load(open("word_freq_dict.p","rb")) # frequency of each word
    class_freq = 0
    counter = 0
    for word in class_dict[c]: # all the words that c has in it:
        counter += 1
        print "making class prob dict.... processed " + str(counter) + " words in class " + c + " of " + str(len(class_dict[c])) + " words in this class overall"
        class_freq += word_freq_dict[word] / float(len(word_dict[word]))  # word's freq / number of classes that w falls into
    return class_freq

"""
We also store all this in a dict so it's less cumbersome.

"""
def get_class_freq_dict():
    class_freq_dict = defaultdict(float)
    class_dict = pickle.load(open("class_dict.p", "rb"))# for each class, its words
    for c in class_dict.keys():
        class_freq_dict[c] = get_class_freq(c)
    return class_freq_dict



"""
Probability of a class c
Next, we get the probability of each class as described by Resnik 1993 (Dissertation) p. 28 equation 2.13

the probability of a given class c is given as: 
#freq(c) / N
#where N is the sum for all c of freq(c)
"""

def get_class_prob(c):
    class_freq_dict = pickle.load(open("class_freq_dict.p", "rb")) # for each class, its frequency
    N = 0
    for k in class_freq_dict.keys():
        print class_freq_dict[k]
        N += class_freq_dict[k]
    class_prob = class_freq_dict[c] / float(N)
    return class_prob
               

"""
Next we make a dict that stores, for each class, its overall probability.
"""
def get_class_prob_dict():
    class_prob_dict = defaultdict(float)
    class_dict = pickle.load(open("class_dict.p", "rb")) # for each class, its words
    counter = 0
    for c in class_dict.keys():
        counter += 1
        print "making class prob dict: processing " + str(counter) + " classes of total" + str(len(class_dict.keys()))
        class_prob_dict[c] = get_class_prob(c)
    return class_prob_dict


"""
As a sanity check, we ensure that the probability of ALL classes adds up to 1.

"""
def sum_up_class_probs():
    class_dict = pickle.load(open("class_dict.p", "rb")) # for each class, its words
    total = 0
    for k in class_dict.keys():
        total += get_class_prob(k)
    return total


"""
Frequency of a class c as the object of a verb v
Next, we get the frequency at which a member of c 
serves as the object of a verb v, as in Resnik 1993 (Dissertation) p. 28 equation 2.14
We don't actually call this, though, we call the dict based on it.
"""
def get_freq_c_with_v(v, c):
    class_dict = pickle.load(open("class_dict.p", "r")) # for each class, its words
    word_dict = pickle.load(open("word_dict.p", "r"))
    verb_obj_dict = pickle.load(open("verb_obj_dict.p", "r")) # frequency of each verb-object pair
    freq = 0 
    for w in class_dict[c]: # for each word in the class c
        freq += (verb_obj_dict[v][w] / float(len(word_dict[w])))
            # number of times w appears as object of v, 
                    #divided by number of classes w falls into
    return freq


"""
We store all that in a dict so it's less cumbersome.
This stores, for every verb, the frequency of every class as an argument of that verb.
....the frequency at which a member of c  serves as the object of a verb v,
as in Resnik 1993 (Dissertation) p. 28 equation 2.14

"""
def get_verb_class_freqs_dict():
    class_dict = pickle.load(open("class_dict.p", "r")) # for each class, its words
    word_dict = pickle.load(open("word_dict.p", "r"))
    verb_obj_dict = pickle.load(open("verb_obj_dict.p", "r")) # frequency of each verb-object pair
    verb_class_freqs_dict = defaultdict(lambda: defaultdict(float))
    for v in verb_obj_dict.keys():
        for c in class_dict.keys():
            freq = 0
            for w in class_dict[c]: # iterates through the words in a class c...
                freq += verb_obj_dict[v][w] / float(len(word_dict[w]))
                    # number of times w appears as object of v, 
                    #divided by number of classes w falls into
            verb_class_freqs_dict[v][c] = freq
    return verb_class_freqs_dict

"""
Probability of a class c given v
Next, we get the probability of a member of class c serving as the object of a given verb v,
from (my understanding of) Resnik 1996 (Cognition paper) p. 137.
"""

## next we get the probability of a class c with verb v
def get_verb_class_prob(v, c):
    verb_obj_dict = pickle.load(open("verb_obj_dict.p", "r")) # for each verb, frequency of each object with it
    verb_class_freqs_dict = pickle.load(open("verb_class_freqs_dict.p", "r"))
                                        # for each verb, its frequency with objects in each class
    N = 0 
    for obj in verb_obj_dict[v].keys():
        N += verb_obj_dict[v][obj]
        # summing up the number of times each obj appears as the object of v
    return verb_class_freqs_dict[v][c] / float(N)



"""
We also store all this in a dict so it's less cumbersome.
For each verb, then we get...
for each class, its probability as an argument of that verb.
"""
def get_verb_class_prob_dict(verbs):
    verb_class_prob_dict = defaultdict(lambda: defaultdict(float))
        # for each verb, the probability of each class as an object of that verb.
    class_dict = pickle.load(open("class_dict.p", "r")) # for each class, its words
    verb_counter = 0
    for verb in verbs:
        verb_counter += 1
        class_counter = 0
        for c in class_dict.keys():
            class_counter += 1
            print "processing verb class probability dict... so far done " + str(class_counter) + " classes for " + str(verb_counter) + " verbs"
            verb_class_prob_dict[verb][c] = get_verb_class_prob(verb, c)
    return verb_class_prob_dict



"""
As a sanity check, we enusre that the
TOTAL probability of ALL classes as the object of a given verb "v" sums to 1.
"""
def sum_up_c_given_v_probs(v):
    class_dict = pickle.load(open("class_dict.p", "rb")) # for each class, its words
    total = 0
    for k in class_dict.keys():
        total += get_verb_class_prob(v, k)
    return total


"""
Selectional strength of a given verb.....
Next, we want the Selection Strength of a given verb v, as defined in Resnik 1996
(Cognition paper) p. 136. These numbers are required to be positive.

I am a bit worried that these numbers might be weirdly low?,
since Resnik reports numbers around 4-6 for "eat" and so on.
But I am hoping that this is just a problem with the size of our toy corpus....? Time will tell.

This is also the part that takes the longest time, because there are SO MANY taxonomic classes.
I have tried to make it slightly more manageable by using "pickle" to store dictionaries
instead of having to re-do calculations each time through the loop.
but obviously, there's a lot of room for improvement here
and it still runs REALLY slowly.
"""
def get_selection_strength(v):
    class_dict = pickle.load(open("class_dict.p", "rb")) # for each class, its words
    class_prob_dict = pickle.load(open("class_prob_dict.p", "rb")) # for each class, its overall probability
    verb_class_prob_dict = pickle.load(open("verb_class_prob_dict.p", "rb"))
                    # for each verb, the probability of each class as object of that verb
    sel_strength = 0
    counter = 0
    for c in class_dict.keys():
        counter += 1
        print "processing selectional strength of " + v + ", \
                    working on class number " + str(counter) + " of " + str(len(class_dict.keys()))
        c_given_v = verb_class_prob_dict[v][c] + 0.000001
       # print c + " given verb (" + v + ") " + str(c_given_v)
        pr_c = class_prob_dict[c] + 0.000001
      #  print "pr of " + c + " overall" + str(pr_c)
        sel_strength += (c_given_v  * math.log(c_given_v / pr_c))
      #  print "sel strength" + str(round(sel_strength, 3))
    if sel_strength < 0:
        print "error, negative selection..."
        print v
        print sel_strength
    return round(sel_strength, 3)


"""
Selectional association
Next, we want the selectional association between a verb and a class,
as defined in Resnik 1993 (Dissertation) p. 80 equation 3.5 and 3.6

These numbers are allowed to be negative as well as positive.
A positive number means that the class is MORE likely given "v" than otherwise.
A negative number means that the class is LESS likely given "v" than otherwise.
We currently don't actually call this function, so don't worry too much about it.
"""

def get_selection_assn(v, c):
    # prob of c given v times log (pr c given v / pr c) is the numerator...
    # divided by sel strength of v....
    c_given_v = get_verb_class_prob(v, c) 
    pr_c = get_class_prob(c)
    #
    sel_strength = get_selection_strength(v)
    return (c_given_v  / sel_strength) * math.log(c_given_v / pr_c)



"""
Finally, we want a dictionary
that provides, for each verb, its selectional strength
(calculated according to the Resnik formulation.)
"""
def get_resnik_sel_dict(verb_obj_pairs):
    get_class_and_word_dicts(verb_obj_pairs)
    print "making class freq dict...."
    class_freq_dict = get_class_freq_dict() # for each class, its frequency. we only make this ONCE.
    pickle.dump(class_freq_dict, open("class_freq_dict.p", "wb")) # then we pickle it.
    
    print "making class prob dict....."
    class_prob_dict = get_class_prob_dict() # for each class, its overall probability. we only make this ONCE
    pickle.dump(class_prob_dict, open("class_prob_dict.p", "wb")) # then we pickle it.

    verb_obj_dict = pickle.load(open("verb_obj_dict.p", "rb")) # for each verb, the frequency of each obj with that verb
    counter = 0
    
    print "making verb class freqs dict...."
    verb_class_freqs_dict = get_verb_class_freqs_dict()
    pickle.dump(verb_class_freqs_dict, open("verb_class_freqs_dict.p", "wb")) # then we pickle it.


    counter = 0
    print "making verb-class prob dict...."
    verb_class_prob_dict = get_verb_class_prob_dict(verb_obj_dict.keys())
    pickle.dump(verb_class_prob_dict, open("verb_class_prob_dict.p", "wb")) # then we pickle it.
    
    # now we make the dict that we're returning....
    resnik_sel_dict = defaultdict(float)

    for verb in verb_obj_dict.keys():
        counter += 1
        print "computing selection of each verb...." + str(counter) + verb + " of " + str(len(verb_obj_dict.keys())) + " total"
        resnik_sel_dict[verb] = get_selection_strength(verb)
    return resnik_sel_dict







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
......... its selectional strength calculated by the Resnik formulation
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
        #d["resnik_sel"] = resnik_dict[verb] 
       
        d["simple_sel"] = simple_sel_dict[verb]["sel"]
        d["most_common_objs"] = simple_sel_dict[verb]["maxobjs"]
        
        d["pmw_count"] = get_pmw(verbdict[verb]["overall_occurrences"], total_word_count)

        # perhaps put this into a different file?
        d["objectless_sentences"] = verbdict[verb]["objectless_sentences"]
        d["sents_WITH_obj"] = verbdict[verb]["sents_WITH_obj"]
        list_of_dicts.append(d)
    dicts_to_csv(list_of_dicts, output_csv)




if __name__ == '__main__':
    txt_csv_to_csv("toy.txt", "toy_csv.csv", "toy_output.csv")
    #this works!! (if you want to edit stuff, do it with "toy.txt" and "toy.csv")
  
    # remove_nonascii("business.txt", "clean_business.txt")
    # cull_relevant_sentences("clean_business.txt", "culled_business.txt", "deduped_trans_verbs.csv")
    #print parse_sentence("They drank water.")



    txt_csv_to_csv("culled_askreddit.txt", "deduped_trans_verbs.csv", "askreddit_output.csv")
    # txt_csv_to_csv("brown_relevant_sentences.txt", "deduped_trans_verbs.csv", "brown_output.csv")
    # txt_csv_to_csv("culled_business.txt", "deduped_trans_verbs.csv", "business_output.csv")


