import nltk
import numpy as np

def create_occurrences_table(global_words,words,occurrences,tags,laplace_parameter):
    """
    Crates an array with length equal to the number of questions. Each of its entries, corresponding to a question, is
    an array of size (1,2) composed by (i) an array of word occurrences in such questions, and (ii) the question's
    label.
    :param global_words: Set of all words found during training.
    :param words: Table of (unique) words per question.
    :param occurrences: Table of corresponding numbers of occurrences of words for each question.
    :param tags: Dictionary with tags and corresponding lists of questions.
    :return: Table (see description above)
    """
    occ_table=[]
    for t in range(len(words)):  # len(words) is equal to the number of questions
        label=[k for k, v in tags.items() if t in v]
        occ_table.append([0]*(len(global_words)+1))
        occ_table[t][-1]=label[0]  # -1 refers to label!!
        for i in range(len(global_words)):
            if global_words[i] in words[t]:
                index=np.where(words[t]==global_words[i])[0]
                occ_table[t][i]=occurrences[t][index][0]
                # Uncomment line below to change occurrences to sqrt
                # occ_table[t][i] = np.sqrt(occurrences[t][index][0])
                # Uncomment line below to change nonzero occurrences to 1
                occ_table[t][i]=(occ_table[t][i]>0)
    # Laplace correction:
    ind=len(words)
    for tag in tags:
        occ_table.append([1]*(len(global_words)+1))
        occ_table[ind][-1]=tag
        ind=ind+1

    #print(occ_table)
    return occ_table

def get_label_probabilities(occ_table,tags):
    num_questions = len(occ_table)
    num_labels = len(tags)
    prob=[0]*num_labels
    i=0
    for tag in tags:
        prob[i]=len(tags[tag])/num_questions
        i=i+1
    #print(prob)
    return prob

def get_occurrence_probabilities(occ_table,tags):
    num_occurrences = len(occ_table[0]) - 1
    num_questions = len(occ_table)
    prob=np.zeros(num_occurrences)
    for i in range(num_occurrences):
        for j in range(num_questions):
            prob[i]=prob[i]+occ_table[j][i]
        prob[i]=prob[i]/num_questions
    #print(prob)
    return prob


def get_occurrence_label_probabilities(occ_table,tags):
    num_occurrences=len(occ_table[0])-1
    num_labels=len(tags)
    prob=np.zeros((num_occurrences,num_labels))
    for i in range(num_occurrences):
        j=0
        num_questions_with_word=0
        for l in range(len(occ_table)):
            num_questions_with_word = num_questions_with_word + occ_table[l][i]
        for tag in tags:
            count=0
            for l in range(len(occ_table)):
                if occ_table[l][-1]==tag:
                    count=count+occ_table[l][i]
            prob[i][j]=count/num_questions_with_word
            j=j+1
        #print(prob[i])
    #print(prob)
    return prob

def get_label_probabilities_from_occurrences(words_test,o_prob,ol_prob,l_prob,global_words,tags):
    """
    This function gives the probability of each label for a given entry; the maximum one corresponds to the predicted
    label.
    :param words_test: Words obtained from processing the entry of test questions.
    :param ol_prob: Occurrence/label probability table.
    :param l_prob: Label probability table
    :return: Label probability vector.
    """
    num_labels=len(l_prob)
    num_questions_test=len(words_test)
    prob=np.ones((num_questions_test,num_labels))
    for i in range(num_questions_test):
        for j in range(len(tags)):
            prob[i][j]=prob[i][j]*l_prob[j]
            for word in words_test[i]:
                if word in global_words:
                    index=np.where(global_words==word)[0][0]
                    prob[i][j]=prob[i][j] * ol_prob[index][j]
                    prob[i][j]=prob[i][j] / o_prob[index]
    print(prob)
    return prob

def predict_labels(tags,lo_probs):
    for i in range(lo_probs.shape[0]):
        j=np.argmax(lo_probs[i])
        k=0
        while k<=j:
            for tag in tags:
                if k==j:
                    print(tag)
                k=k+1
