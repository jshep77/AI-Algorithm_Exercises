#!/usr/bin/env python3
###################################
# CS B551 Fall 2018, Assignment #3
#
# Your names and user ids: 
#
# (Based on skeleton code by D. Crandall)
#
#
####
# Put your report here!!
###*****Disclaimer*****
#The code takes a long time to run due to the high number of samples taken in 
#MCMC sampling algorithm

##The variables used in training are-
#Initial probability(ps1)
# This is probability of each tag occuring at first position. It is the frequency 
# that each tag at first position by the total count. The values are stored in
#'log' values since all calculations are performed in log to avoid underflow
#
#State(ps)
#This is the probability of a tag occurring irrespective of the position except
#for the tags occuring at the first position. It is frequency of each tag divided
#by total number of tags. The values are stored in log 
#
#Transition(pss)
#This is probability of two tags occurring in sequence. It is frequency of all
#tag pairs in sequence divided by total number of tags pairs. The values are 
#stored in log. pss variable is a dictionary which contain pair of tags as pairs
#e.g.P[('tag1','tag2')]. 
#Please note that ‘pss’ stores the probabilities that two tags occur in a sequence.
#This is P(S and Si+1). To find P(Si+1|Si) we have to subtract P(Si) from P(S and Si+1).
#
#Emission(pws)
#The is the probability of each word given a tag. For all tags, frequency is calculated
#for associated words, the probability is calculated by diving the frequency of 
#each word by total number of words associated with that tag. The values are stored
#in log. 
#
#Complex(pc)
#This is the probability that three tags occur in a sequence. The values are 
#stored in log.
#This is P(S,Si+1,Si+2). To find P(Si+2|Si,Si+1) we have to subtract P(Si,Si+1) 
#from P(S,Si+1,Si+2).
#This probability is calculated using the function ‘joint_probability’  
#
##############################################################################
#ALGORITHMS
##############################################################################
#
#NAIVE BAYES(SIMPLE)
#This is implemented by maximising the posterior probability(check posterior
#probability section) for each tag given a word. In case the test data has an
#unseen word. A small probability is substituted for P(Word|Tag)
#
#VITERBI(HMM)
#The algorithm is implemented by initializing two  dictionary of dictionaries
#'viterbi' and 'backtrack' the keys of outer dictionary are the unique tags 
#and the keys of the inner dict are the words in the sentence. The viterbi dict
#stores the max score obtained by implementing the viterbi algorithm at a 
#particular state and the backtrack dict stores the value of tags from which
#the maximum score is obtained(argmax). To generate a sequence the tag is obtained
#from the viterbi dict whose probability is maximum for the last word, this is 
#the last tag of the given sentence. corresponding values is obtained from 
#backtrack dict, this is the tag of the the second last word. The tags are traced
#back from the backtrack dict. This order is then reversed to obtain the sequence
#of tags for the given sentence. If in test a word does not appear for a given
#tag then it is replaced by a samll probability
#
#MCMC(COMPLEX)
#For a given sentence a random sequence of tags is initialized, from this sequence
#for each position, starting from first tag, the values are changed to each tag
#and the corresponding joint probability is calculated using 'joint_probability'
#function keeping all the other tags fixed. Then a random tag is fixed at the 
#position using the function random.choice, then same step is performed for all
#the tags in the tag sequence, this way one sample is generated. Number of iterations
#and burning iterations(samples to discard initially) are chosen, this creates
#number of samples from the given probability distribution. To find out the sequence
#from the samples, we have chosen the tag appearing the most number of times at each
#position
#
##Posterior probabilities
#
#[Simple]
#The posterior probability is calculated using Naive Bayes assumption, that
#each hidden variable is independent from every other hidden variable. Given a 
#set of observed variables(words) and hidden variables (POS tags), for each word 
#and corresponding tag, the posterior probability of each tag given a word is given by 
#P(tag|word) = (P(word|tag) * P(tag))/P(word)
#We can ignore the denominator as we have to compare the score for each tag and
#consider the maximum.
# i.e. P(tag|word) α (P(word|tag) * P(tag))
#This is for one word and corresponding tag, for a given sequence, we have to 
#multiply the posterior probabilities of all the word-tag combination
#P(tag1|word1)*P(tag2|word2)…P(tagn|wordn) α (P(word1|tag1) * P(tag1)) * 
#                       (P(word2|tag2) * P(tag2)) ….. (P(wordn|tagn) * P(tagn))
# Also, since the probabilities are stored in logs, we convert the above equation
# to the following by taking logs on both sides
#Posterior probability of the given word-tag sequence = log(P(tag1|word1)) + 
#                                       log(P(tag2|word2))…+log(P(tagn|wordn))
#In the program the probability P(word|tag) is the emission probability(likelihood)
# and is stored in the variable ‘pws’ and the the probability P(tag) is the probability
# of the hidden variable(prior) and is stores in the variable ‘ps’
#
#[HMM]
#The probability of each element in a Hidden Markov Chain depend on the previous
#state(transition probability) and the probability of each word(emission probability)
#depends on the hidden state. The probability of a given sequence of words and 
#hidden states(tags) is given by
#Probaility = [P(tag1)*P(word1|tag1)]* [P(word2|tag2)*P(tag2|tag1)]….* 
#                                               [P(wordn|tagn)*P(tagn|tagn-1)]
#The transition probabilities (e.g. P(tag2|tag1)) are stored in pss
#
#[Complex]
#According to figure 1c given in the question, the probability of state depends
# on the two previous states(complex probability) and the probability of each 
#word(emission probability) depends on the hidden state. The probability of a 
#given sequence of words is given by
#Probability = [P(tag1)*P(word1|tag1)]* [P(word2|tag2)*P(tag2|tag1)]*
#       [P(word3|tag3)*P(tag3|tag2,tag1)]…* [P(wordn|tagn)*P(tagn|tagn-1,tagn-2)]
#The complex probabilities (e.g. P(tag3|tag1,tag2))  are stored in ‘pc’
#Please note that ‘pc’ stores the probabilities that three tags occur in a sequence.


####

import random
import math
from collections import Counter
import pandas as pd
import numpy as np
import operator
from copy import deepcopy
import itertools


# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    pws, pss, ps, ps1 = 0,0,0,0
    tags_uniq = 0
    small_prob=math.log(1/100000000)
    # Calculate the log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
        
    def posterior(self, model, sentence, label):
        words = list(sentence)
        labels = list(label)
        if model == "Simple":
#            print('model',model)
#            print('sentence',sentence)
#            print('label',label)
            
            prob=0
            for word, label in zip(words, labels):
                if word in self.pws.index:
                    prob += self.pws.loc[word][label]+self.ps[label]
                else:
                    prob += self.small_prob+self.ps1[label]
            return prob
        elif model == "Complex":
            prob=0
            #For position=0
            prob = self.joint_probability(labels,words)
            if prob == 0:
                prob = 1/100000000
            return math.log(prob)
            return -999
        elif model == "HMM":
            prob=0
            if words[0] in self.pws.index:
                prob=self.pws.loc[words[0]][labels[0]]+self.ps1[labels[0]]
            else:
                prob=self.small_prob+self.ps1[labels[0]]
            last_label = labels[0]
            #print('ll:',last_label)
            
            if len(words)>=2:
                if words[1] in self.pws.index:
                    prob+=self.pws.loc[words[1]][labels[1]]+self.pss[(labels[0],labels[1])]-self.ps1[labels[0]]
                else:
                    prob=self.small_prob+self.pss[(labels[0],labels[1])]-self.ps1[labels[0]]
                last_label = labels[1]
            #print('ll:',last_label)
            
            if len(words)>=3:
                for i, j in zip(range(2,len(words)),range(2,len(labels))):
                    if words[i] in self.pws.index:
                        prob+=self.pws.loc[words[i]][labels[j]]+self.pss[(last_label,labels[j])]-self.ps[last_label]
                    else:
                        prob+=self.small_prob+self.pss[(last_label,labels[j])]-self.ps[last_label]
                    last_label = labels[j]
                #print('ll:',last_label)
            #print('hmm prob:',prob)         
            return prob
        else:
            print("Unknown algo!")

    # Do the training!
    #
    def train(self, data):
        tags1=[]
        for i in range(len(data)):
            #print(i)
            tags1.append(data[i][1][0])
        
        self.tags_uniq = list(set(tags1))
        
        #######################################################################
        #Initial probability(ps1)
        #######################################################################
        #P(S1)
        self.ps1=dict(Counter(tags1))
        self.ps1 = {k: math.log(v / total) for total in (sum(self.ps1.values(), 0.0),) for k, v in self.ps1.items()}
        
        #######################################################################
        #State probability(ps)
        #######################################################################
        #P(S)
        tags3=[]
        for i in range(len(data)):
            t = list(data[i][1][1:])
            #print('asd:',t)
            tags3.append(t)
            #print(tags3)
        
        tags4 = [item for sublist in tags3 for item in sublist]
        self.ps=dict(Counter(tags4))
        self.ps = {k: math.log(v / total) for total in (sum(self.ps.values(), 0.0),) for k, v in self.ps.items()}
        
        #######################################################################
        #Transition probability(pss)
        #######################################################################
        #P(Si+1|Si)
        self.pss={}
        tags2=[]
        for i  in range(len(data)):
            for j in range(1,len(data[i][1])):
                t1=data[i][1][j]
                t2=data[i][1][j-1]
                t=(t2,t1)
                tags2.append(t)
        self.pss=dict(Counter(tags2))
        self.pss = {k: math.log(v / total) for total in (sum(self.pss.values(), 0.0),) for k, v in self.pss.items()}
        self.pss[('pron', 'x')]=math.log(1/1000000)
        
        #######################################################################
        #Emission probability(pws)
        #######################################################################        
        #P(Wi|Si)
        self.pws={}
        for tag in self.tags_uniq:
            l=[]
            for i in range(len(data)):
                for j in range(len(data[i][1])):
                    #print(t)
                    if data[i][1][j]==tag:
                        #print('dani california')
                        l.append(data[i][0][j])
            l=dict(Counter(l))
            l = {k: math.log(v / total) for total in (sum(l.values(), 0.0),) for k, v in l.items()}
            self.pws[tag]=l
        self.pws=pd.DataFrame(self.pws)
        self.pws = self.pws.fillna(math.log(1/1000000))
        
        #######################################################################
        #Complex Probability(pc)
        #######################################################################
        #P(Si|Si-2)
        tags3=[]
        for i in range(len(data)):
            for j in range(2,len(data[i][1])):
                t1=data[i][1][j]
                t2=data[i][1][j-1]
                t3=data[i][1][j-2]
                t=(t3,t2,t1)
                tags3.append(t)
        self.pc=dict(Counter(tags3))
        self.pc = {k: math.log(v / total) for total in (sum(self.pc.values(), 0.0),) for k, v in self.pc.items()}
                             
        #print('hey ho!',self.pss[('det','noun')]-self.ps['det'])
       
        pass
    # Functions for each algorithm. Right now this just returns nouns -- fix this!
    #
    def simplified(self, sentence):
        words=list(sentence)
        #print(words)
        tags=[]
        scores=[]
        for word in words:
            max_tag_score = -10000
            max_tag = ''
            if word in self.pws.index:
                for tag in self.tags_uniq:
                    tag_score=self.ps[tag]+self.pws.loc[word][tag]
                    if tag_score > max_tag_score:
                        max_tag_score = tag_score
                        max_tag = tag
            #print(word,max_tag,max_tag_score)
            else:
                max_tag=max(self.ps.items(), key=operator.itemgetter(1))[0]
             
            tags.append(max_tag)
        #print('Under the bridge',tags)
        return tags

    #Function to calculate joint probability
    #This function takes the given words(emission variables) and assumed states
    #and calculates the probability of the given sequence of part of speech tags
    #and given words based on the complex structure given in Fig 1c
    def joint_probability(self, tags, words):
        joint_prob=0
        
        #For first terms
        joint_prob = self.pws.get(tags[0],{}).get(words[0],self.small_prob)+self.ps1[tags[0]]
        
        #For 2nd terms
        if len(words)>=2:
            joint_prob += self.pss.get((tags[0],tags[1]),self.small_prob)-self.ps1[tags[0]]+self.pws.get(tags[1],{}).get(words[1],self.small_prob)
        
        #For subsequent terms
        if len(words)>=3:
            for i, j in zip(range(2,len(words)),range(2,len(tags))):
                joint_prob += self.pc.get((tags[j-2],tags[j-1],tags[j]),self.small_prob)-self.pss.get((tags[j-2],tags[j-1]),self.small_prob)+self.pws.get(tags[j],{}).get(words[i],self.small_prob)
        
        return math.exp(joint_prob)
    
    def complex_mcmc(self, sentence):
        words = list(sentence)
        sample = [ "noun" ] * len(sentence)
        itr = 50
        burn_itr = 20
        samples=[]
        for n in range(0,itr):
            new_sample = sample[:]
            #print('new sample:',new_sample)
            tags = new_sample[:]
            sample=[]
            for i in range(len(new_sample)):
                e_dict={}
                e_dict = e_dict.fromkeys(self.tags_uniq)
                for tag in self.tags_uniq:
                    tags[0:len(sample)] = sample
                    tags[i] = tag
                    #tags = sample+[tag]+new_sample[i+1:]
                    #print('tags',tags)
                    prob=self.joint_probability(tags, words)
                    e_dict[tag] = prob
                try:
                    e_dict = {k: v / total for total in (sum(e_dict.values(), 0.0),) for k, v in e_dict.items()}
                    choice = np.random.choice(list(e_dict.keys()),1,p=list(e_dict.values()))
                except:
                    e_dict = {k: v / self.small_prob for k, v in e_dict.items()}
                    choice = np.random.choice(list(e_dict.keys()),1)
                choice = choice[0]
                sample.append(choice)
                #print('sample:',sample)
            if n>burn_itr:
                samples.append(sample)
        #print(len(samples),samples[0])
        seq=[]
        for i in range(len(samples[0])):
            pos_elements=[]
            for j in range(len(samples)):
                pos_elements.append(samples[j][i])
                c = Counter(pos_elements)
                most_common = c.most_common(1)[0][0]
            seq.append(most_common)
        #print('seq:',seq)
        return seq 
        #return [ "noun" ] * len(sentence)           


    def hmm_viterbi(self, sentence):
        words=list(sentence)
        viterbi={}
        viterbi = viterbi.fromkeys(self.tags_uniq)
        backtrack={}
        backtrack = backtrack.fromkeys(self.tags_uniq)
        for tag in self.tags_uniq:
            words_dict={}
            words_dict = words_dict.fromkeys(words)
            viterbi[tag] = deepcopy(words_dict)
            if words[0] in self.pws.index:
                pws=self.pws.loc[words[0]][tag]
            else:
                pws=self.small_prob
                #pws = max(self.ps1.items(), key=operator.itemgetter(1))[1]
            ps1=self.ps1[tag]
            #print('ps1',ps1)
            viterbi[tag][words[0]]=pws+ps1
            backtrack[tag] = deepcopy(words_dict)
            backtrack[tag][words[0]]=0
        #print('viterbi',viterbi)
        #print('backtrack:',backtrack)
        for i in range(1,len(words)):
            for tag_c in self.tags_uniq:
                if words[i] in self.pws.index:
                    emission = self.pws.loc[words[i]][tag_c]
                else:
                    emission=self.small_prob
                    #emission = max(self.ps.items(), key=operator.itemgetter(1))[1]
                max_prob=-10000
                max_tag=''
                for tag_p in self.tags_uniq:
                    #print('pss:',self.pss[(tag_p,tag_c)])
                    #print('vit',viterbi[tag_p][words[i-1]])
                    prob = self.pss[(tag_p,tag_c)]-self.ps[tag_p]+viterbi[tag_p][words[i-1]]
                    #print('prob:',prob)
                    if prob > max_prob:
                        max_prob = prob
                        max_tag = tag_p
                        #print('max_tag',max_tag)
                viterbi[tag_c][words[i]]=max_prob+emission
                backtrack[tag_c][words[i]]=max_tag
                #print('viterbi',viterbi)
                #print('backtrack:',backtrack)
        max_prob = -10000
        max_tag=''
        tag_return=[]
        for tag in self.tags_uniq:
            #print(words[-1])
            last_word = words[-1]
            prob = viterbi[tag][last_word]
            if prob > max_prob:
                max_prob = prob
                max_tag = tag
        tag_return.append(max_tag)
        for word in reversed(words[1:]):
            #print('word:',word)
            #print('max_tag:',max_tag)
            tag_return.append(backtrack[max_tag][word])
            max_tag=backtrack[max_tag][word]
        
        tag_return = tag_return[::-1]    
        #print('tag_return:',tag_return)          
        #return [ "noun" ] * len(sentence)
        return tag_return
        


    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself. 
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, sentence):
        if model == "Simple":
            return self.simplified(sentence)
        elif model == "Complex":
            return self.complex_mcmc(sentence)
        elif model == "HMM":
            return self.hmm_viterbi(sentence)
        else:
            print("Unknown algo!")