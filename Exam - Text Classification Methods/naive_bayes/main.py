import numpy as np
import nltk
import unicodedata
import naive_bayes
from collections import OrderedDict

def normalize_word(word):
    word=''.join(ch for ch in unicodedata.normalize('NFKD', word)
                 if not unicodedata.combining(ch))
    # The lines above eliminate diacritics form the word
    word = word.lower()
    word = word.rstrip(".")
    word = word.rstrip("(")
    word = word.rstrip(")")
    word = word.rstrip("´")
    word = word.rstrip("~")
    word = word.rstrip("^")
    word = word.rstrip("¸")
    word = word.rstrip("`")
    word = word.rstrip(",")
    word = word.rstrip("/")
    word = word.rstrip("+")
    word = word.rstrip("?")
    word = word.rstrip("-")
    word = word.rstrip(":")
    word = word.rstrip(";")
    stemmer = nltk.stem.RSLPStemmer()
    if len(word)>0:
        word=stemmer.stem(word)
    return word

def process_text_entry(filename):
    """
        Reads a document .txt and, for each question, returns a list of unique words and another of numbers of
        occurrences of those words. Also returns a list of all words appearing at least once in the examples of
        questions.
        Question texts are previously separated by newlines.
        """
    file = open(filename, "r")
    i = 0
    fields = []
    for question in file:
        fields.append(question.split(" "))
        fields[i][-1] = fields[i][-1].rstrip("\n")  # Removing final "\n" from line.
        i = i + 1

    # print(fields)
    i = 0
    for question in fields:
        j = 0
        for word in question:
            word = normalize_word(word)
            fields[i][j] = word
            j = j + 1
        fields[i].sort()
        i = i + 1

    words=[]
    occurrences=[]
    fields_=[]
    for x in range(len(fields)):
        fields_.append(np.array(fields[x]))
        words.append(np.unique(fields_[x]))
        occurrences.append(np.zeros(len(words[x])))
        for y in range(len(words[x])):
            occurrences[x][y]=fields[x].count(words[x][y])
            # Uncomment or comment:
            occurrences[x][y]=(occurrences[x][y]>0)

    global_words=[]
    for element in words: #in fields
        for word in element:
            global_words.append(word)
    # print(words)
    global_words=np.array(global_words)
    #for t in range(len(global_words)):
       # global_words[t]=normalize_word(global_words[t])
    global_words=np.unique(global_words)
    global_words.sort()
    print(global_words)
    print(words)
    #print(occurrences)
    return global_words,words,occurrences


def process_tag_entry(filename,laplace_parameter):
    """
    Receives the tags corresponding to the questions (same order) and returns (...)
    :param filename: name of input file with tags.
    :type filename: str
    :return:
    """
    tags=OrderedDict()

    file=open(filename, "r")
    i=0
    fields=[]
    for tag in file:
        fields.append(tag)
        fields[i]=fields[i].rstrip("\n")
        i=i+1
    for x in range(len(fields)):
        if not fields[x].lower() in tags:
            tags[fields[x].lower()]=[x]
        else:
            tags[fields[x].lower()].append(x)
    # Laplace correction:
    if laplace_parameter>0:
        for tag in tags:
            tags[tag].append(i)
            i=i+1

    for key, val in tags.items():
        print(key, "=>", val)
    return tags





laplace_correction_parameter=1
global_words,words,occurrences=process_text_entry("Questions.txt")
tags=process_tag_entry("Tags.txt",laplace_correction_parameter)
occ_table=naive_bayes.create_occurrences_table(global_words,words,occurrences,tags,laplace_correction_parameter)
l_prob=naive_bayes.get_label_probabilities(occ_table,tags)
ol_prob=naive_bayes.get_occurrence_label_probabilities(occ_table,tags)
o_prob=naive_bayes.get_occurrence_probabilities(occ_table,tags)

global_words_test,words_test,occurrences_test=process_text_entry("Questions_.txt")
lo_probs=naive_bayes.get_label_probabilities_from_occurrences(words_test,o_prob,ol_prob,l_prob,global_words,tags)
naive_bayes.predict_labels(tags,lo_probs)