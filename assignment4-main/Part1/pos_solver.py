###################################
# CS B551 Fall 2022, Assignment #4
#
# Your names and user ids: 
#Name: Venkata Naga Sreya Kolachalama | user id : vekola
#Name: Joseph Shepherd | user id: joseshep
#
# (Based on skeleton code by D. Crandall)
#
import random
from math import log
from collections import Counter
import numpy as np
# We've set up a suggested code structure, but feel free to change it. Just
# make sure your code still works with the label.py and pos_scorer.py code
# that we've supplied.
#
class Solver:
    def __init__(self):
        # Calculating transition probability
        self.transition_prob = { }
        # Calculating initial state distribution of tag
        self.initial_prob = { }
        # Calculating initial distribution of words
        self.initial_word = { }
        # Calculating emission probability
        self.emission_prob = { }
        # Calculating next next transition probability for complex model
        self.successor_transition_prob = { }
        # Calculating next emission probability for complex model        
        self.succesor_emission_prob = { }       
        self.tagcounts_for_words = {}  # Dictionary to store tag counts for words
        self.total_tag_count = {}  # Dictionary to store total tag counts
    # Do the training!
    #
    def train(self, data):
        for given_sentence in data:
            for i in range(len(given_sentence[0])):
                word_val = given_sentence[0][i]
                tag_val = given_sentence[1][i]               
                # Calculating emission probility
                if tag_val not in self.emission_prob:
                    self.emission_prob[tag_val] = [word_val]
                else:
                    self.emission_prob[tag_val].append(word_val)  
                if i < len(given_sentence[0])-1:
                    next_tag = given_sentence[1][i+1]
                    next_word = given_sentence[0][i+1]
                # Calculating transition probability
                    if tag_val not in self.transition_prob:
                        self.transition_prob[tag_val] = [next_tag]
                    else:
                        self.transition_prob[tag_val].append(next_tag)
                    if tag_val not in self.succesor_emission_prob:
                        self.succesor_emission_prob[tag_val] = [next_word]
                    else:
                        self.succesor_emission_prob[tag_val].append(next_word)
                # Calculating probability for next next tag
                if i < len(given_sentence[0])-2:
                    next_next_tag = given_sentence[1][i+2]
                    if tag_val not in self.successor_transition_prob:
                        self.successor_transition_prob[tag_val] = [next_next_tag]
                    else:
                        self.successor_transition_prob[tag_val].append(next_next_tag)                   
                # Calculating initial distribution
                if tag_val not in self.initial_prob:
                    self.initial_prob[tag_val] = 1
                else:
                    self.initial_prob[tag_val] += 1
                # Calculating word frequency
                if word_val not in self.initial_word:
                    self.initial_word[word_val] = 1
                else:
                    self.initial_word[word_val] += 1         
        for tag_val in self.initial_prob:
            self.initial_prob[tag_val] /= sum(self.initial_prob.values()) 
        for word_val in self.initial_word:
            self.initial_word[word_val] /= sum(self.initial_word.values())   
        for tag_val in self.emission_prob.keys():
            l = len(self.emission_prob[tag_val])
            count = Counter(self.emission_prob[tag_val])
            self.emission_prob[tag_val] = {i:(j/l) for i,j in count.items()}    
        for tag_val in self.transition_prob.keys():
            l = len(self.transition_prob[tag_val])
            count = Counter(self.transition_prob[tag_val])
            self.transition_prob[tag_val] = {i:(j/l) for i,j in count.items()}
        for tag_val in self.succesor_emission_prob.keys():
            l = len(self.succesor_emission_prob[tag_val])
            count = Counter(self.succesor_emission_prob[tag_val])
            self.succesor_emission_prob[tag_val] = {i:(j/l) for i,j in count.items()}
        for tag_val in self.successor_transition_prob.keys():
            l = len(self.successor_transition_prob[tag_val])
            count = Counter(self.successor_transition_prob[tag_val])
            self.successor_transition_prob[tag_val] = {i:(j/l) for i,j in count.items()}

#****************Simple model*********************

    def Simple_model(self, sentence):       
        result = [ ]
        for word in list( sentence ) :
            p_val = 0.0
            t_val = 'x'
            for tag in self.emission_prob.keys():
                if word in self.emission_prob[tag].keys():
                    q_val = self.emission_prob[tag][word] * self.initial_prob[tag]
                    if q_val > p_val:
                        t_val = tag
                        p_val = q_val          
            result.append(t_val)
        return result
    
 #*****************HMM model**************************   

    def generate(self, given_sentence, label):
        N = len(given_sentence)
        tag_vals = ['adj' ,'adv' ,'adp' ,'conj','det' ,'noun','num' ,'pron','prt' ,'verb','x'  ,'.' ]
        for i in range(N):
            word_val = given_sentence[i]
            prob = [0] * len(tag_vals)    
            for j in range(len(tag_vals)):
                label_vals = tag_vals[j]
                em_prob = self.emission_prob[label_vals].get(word_val, 1e-10)
                if i < N-1:
                    em_prob_next = self.succesor_emission_prob[label_vals].get(given_sentence[i+1], 1e-10)
                    trans_prob = self.transition_prob[label_vals].get(label[i+1],1e-10)
                else:
                    em_prob_next = 1
                    trans_prob = 1  
                if i < N-2:
                    trans_prob_next = self.successor_transition_prob[label_vals].get(label[i+2],1e-10)
                else:
                    trans_prob_next = 1   
                if i > 0:
                    trans_prob_prev = self.transition_prob[label[i-1]].get(label_vals,1e-10)
                else:
                    trans_prob_prev = 1  
                if i > 1:
                    trans_prob_prev_prev = self.successor_transition_prob[label[i-2]].get(label_vals,1e-10)
                else:
                    trans_prob_prev_prev = 1     
                prob[j] = em_prob_next * trans_prob * trans_prob_next * em_prob * trans_prob_prev * trans_prob_prev_prev     
            tot = sum(prob)
            prob = [x / tot for x in prob]
            rand_val = random.random()
            prob_tot = 0
            for k in range(len(tag_vals)):
                prob_tot += prob[k]
                if prob_tot > rand_val:
                    label[i] = tag_vals[k]
                    break     
        return label

    def HMM_Viterbi_model(self, given_sentence):
        tag_vals = ['adj', 'adv', 'adp', 'conj', 'det', 'noun', 'num', 'pron', 'prt', 'verb', 'x', '.']
        N = len(given_sentence)
        tot_tags = len(tag_vals)
        viterbi_matrix = np.zeros((tot_tags, N))
        backpointer_matrix = np.zeros((tot_tags, N), dtype=int)
        initial_word = given_sentence[0]
        for t in range(tot_tags):
            viterbi_matrix[t, 0] = self.initial_prob[tag_vals[t]] * self.emission_prob[tag_vals[t]].get(initial_word, 1e-10)
        for t in range(1, N):
            for s in range(tot_tags):
                # Calculating probability and backpointer for each tag
                prob_vals = [viterbi_matrix[s_prev, t - 1] * self.transition_prob[tag_vals[s_prev]].get(tag_vals[s], 1e-10) * self.emission_prob[tag_vals[s]].get(given_sentence[t], 1e-10) for s_prev in range(tot_tags)]
                backpointer_matrix[s, t] = np.argmax(prob_vals)
                viterbi_matrix[s, t] = max(prob_vals)
        best_soln = [np.argmax(viterbi_matrix[:, -1])]
        for t in range(N - 1, 0, -1):
            best_soln.append(backpointer_matrix[best_soln[-1], t])
        # Reversing the received solution to get the correct order
        best_soln = best_soln[::-1]
        # Converting to tag name
        soln = [tag_vals[i] for i in best_soln]
        return soln

#*******************MCMC model************************* 

    def Complex_MCMC_model(self, given_sentence):
     label = [""] * len(given_sentence)  # Initialize an empty label
     for var in range(len(given_sentence)):
        try:
            label = self.generate(given_sentence, label)  # Pass both given_sentence and label
        except IndexError as e:
            print(f"Error in iteration {var}: {e}")
            print("Given Sentence:", given_sentence)
            print("Current Solution:", label)
            raise  # Reraise the exception to terminate the program
     return label

    # Calculating log of the posterior probability of a given sentence
    #  with a given part-of-speech labeling. Right now just returns -999 -- fix this!
    def posterior(self, model, given_sentence, label):
        if model == "Simple":
            log_post = 0.0
            for i in range(len(given_sentence)):
                word = given_sentence[i]
                lab = label[i]
                log_post += log(self.initial_prob.get(lab, 1e-24)) + log(self.emission_prob[lab].get(word, 1e-24)) - log(self.initial_word.get(word, 1e-24))
            return log_post      
        elif model == "Complex":
            log_post = log(self.initial_prob[label[0]])+log(self.emission_prob[label[-1]].get(given_sentence[-1], 1e-24))
            for i in range(len(given_sentence)-1):
                cur_label = label[i]
                next_label = label[i+1]
                word = given_sentence[i]
                next_word = given_sentence[i+1]
                if cur_label in self.transition_prob and next_label in self.transition_prob[cur_label]:
                    log_post += log(self.transition_prob[cur_label][next_label])
                else: 
                    log_post += log( 1e-24 )
                if cur_label in self.emission_prob and word in self.emission_prob[cur_label]:
                    log_post += log(self.emission_prob[cur_label][word])
                else:
                    log_post += log( 1e-24 )
                if cur_label in self.succesor_emission_prob and next_word in self.succesor_emission_prob[cur_label]:
                    log_post += log(self.succesor_emission_prob[cur_label][next_word])
                else:
                    log_post += log( 1e-24 )
            for i in range(len(given_sentence)-2):
                cur_label = label[i]
                next_next_label = label[i+1]
                if cur_label in self.successor_transition_prob and next_next_label in self.successor_transition_prob[cur_label]:
                    log_post += log(self.successor_transition_prob[cur_label][next_next_label])
                else:
                    log_post += log( 1e-24 )
            return log_post   
        elif model == "HMM":
            log_post = log(self.initial_prob[label[0]])+log(self.emission_prob[label[-1]].get(given_sentence[-1], 1e-24))
            for i in range(len(given_sentence)-1):
                cur_label = label[i]
                next_label = label[i+1]
                word = given_sentence[i]
                if cur_label in self.transition_prob and next_label in self.transition_prob[cur_label]:
                    log_post += log(self.transition_prob[cur_label][next_label])
                else: 
                    log_post += log( 1e-24 )
                if cur_label in self.emission_prob and word in self.emission_prob[cur_label]:
                    log_post += log(self.emission_prob[cur_label][word])
                else:
                    log_post += log( 1e-24 )
            return log_post  
        else:
            print("Unknown algo!")

    # This solve() method is called by label.py, so you should keep the interface the
    #  same, but you can change the code itself.
    # It should return a list of part-of-speech labelings of the sentence, one
    #  part of speech per word.
    #
    def solve(self, model, given_sentence):
        if model == "Simple":
            return self.Simple_model(given_sentence)
        elif model == "Complex":
            return self.Complex_MCMC_model(given_sentence)
        elif model == "HMM":
            return self.HMM_Viterbi_model(given_sentence)
        else:
            print("Unknown algo!")