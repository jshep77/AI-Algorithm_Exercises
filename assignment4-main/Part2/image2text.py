#!/usr/bin/python
#
# Perform optical character recognition, usage:
#     python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png
#
# Authors: (insert names here)
# (based on skeleton code by D. Crandall, Nov 2023)
#

from PIL import Image
import sys
import numpy as np

CHARACTER_WIDTH = 14
CHARACTER_HEIGHT = 25

class HMM:
    def __init__(self, states, observations):
        self.states = states
        self.observations = observations
        self.transition_probs = np.zeros((len(states), len(states)))
        self.initial_state_probs = np.zeros(len(states))
        self.emission_probs = {}

    def train(self, train_text, train_letters):
        self.estimate_transition_probs(train_text)
        self.estimate_initial_state_probs(train_text)
        self.estimate_emission_probs(train_letters)

    def recognize(self, test_letters):
        result = []
        for test_letter in test_letters:
            path, _ = self.viterbi(test_letter)
            result.append(self.decode_path(path))
        return ''.join(result)

    def estimate_transition_probs(self, train_text):
        for i in range(len(train_text) - 1):
            try:
                current_state = self.states.index(train_text[i])
                next_state = self.states.index(train_text[i + 1])
                self.transition_probs[current_state, next_state] += 1
            except ValueError:
                pass

        # Normalize transition probabilities
        if np.sum(self.transition_probs) != 0:
            self.transition_probs = (self.transition_probs / np.sum(self.transition_probs, axis=1, keepdims=True) + 1e-10)
        else:
            pass

    def estimate_initial_state_probs(self, train_text):
        initial_state = train_text[0]
        if initial_state in self.states:
            initial_state_index = self.states.index(initial_state)
            self.initial_state_probs[initial_state_index] += 1

            # Normalize initial state probabilities
            self.initial_state_probs = self.initial_state_probs / np.sum(self.initial_state_probs)
        else:
            pass

    def estimate_emission_probs(self, train_letters):
        for state in self.states:
            if state not in train_letters:
                pass

            reference_letter = train_letters[state][0].strip()  # Assuming the first row of each letter is representative
            for obs in self.observations:
                obs_index = self.observations.index(obs)
                count = sum(row.count('*') for row in reference_letter)
                if count == 0:
                    pass
                else:
                    self.emission_probs[state, obs] = reference_letter[obs_index].count('*') / count

    def viterbi(self, test_letter):
        T = len(test_letter)
        N = len(self.states)

        # Initialization
        delta = np.zeros((N, T))
        psi = np.zeros((N, T))

        # Initialize the first column of delta
        for i in range(N):
            initial_prob = self.initial_state_probs[i] if test_letter[0] in self.emission_probs else 0
            delta[i, 0] = initial_prob * self.emission_probs.get((self.states[i], test_letter[0]), 0)

        # Recursion
        for t in range(1, T):
            for j in range(N):
                delta[j, t] = np.max(delta[i, t - 1] * self.transition_probs[i, j] * self.emission_probs.get((self.states[j], test_letter[t]), 0))

        # Termination
        path = np.zeros(T, dtype=int)
        path[T - 1] = np.argmax(delta[:, T - 1])

        # Backtracking
        for t in range(T - 2, -1, -1):
            path[t] = psi[path[t + 1], t + 1]

        return path, delta

    def decode_path(self, path):
        return ''.join([self.states[i] for i in path])

def load_letters(fname):
    im = Image.open(fname)
    px = im.load()
    (x_size, y_size) = im.size
    result = []
    for x_beg in range(0, int(x_size / CHARACTER_WIDTH) * CHARACTER_WIDTH, CHARACTER_WIDTH):
        result += [
            [
                "".join(['*' if px[x, y] < 1 else ' ' for x in range(x_beg, x_beg + CHARACTER_WIDTH)])
                for y in range(0, CHARACTER_HEIGHT)
            ],
        ]
    return result

def load_training_letters(fname):
    TRAIN_LETTERS = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
    letter_images = load_letters(fname)
    return {TRAIN_LETTERS[i]: letter_images[i] for i in range(0, len(TRAIN_LETTERS))}

def naive_bayes_classifier(test_letter, train_letters):
    # Flatten the training letters into a list of strings
    flat_train_letters = [''.join(row) for letter in train_letters for row in letter]

    # Flatten the test letter into a string
    flat_test_letter = ''.join(test_letter)

    # Calculate the similarity between the test letter and each training letter
    similarities = [sum(a == b for a, b in zip(flat_test_letter, flat_train)) for flat_train in flat_train_letters]

    # Find the index of the most similar training letter
    max_index = similarities.index(max(similarities))

    # Map the index to the corresponding character (A, B, C, etc.)
    recognized_char = chr(max_index + ord('A'))

    return recognized_char


def simple_bayes_recognition(test_letters, train_letters):
    result = []
    for test_letter in test_letters:
        recognized_char = naive_bayes_classifier(test_letter, train_letters)
        result.append(recognized_char)

    return ''.join(result)



#####
# main program
if len(sys.argv) != 4:
    raise Exception("Usage: python3 ./image2text.py train-image-file.png train-text.txt test-image-file.png")

(train_img_fname, train_txt_fname, test_img_fname) = sys.argv[1:]
train_letters = load_training_letters(train_img_fname)
train_text = open(train_txt_fname).read()
test_letters = load_letters(test_img_fname)

# Initialize and train the HMM
states = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789(),.-!?\"' "
observations = ["".join(row) for row in train_letters]
hmm = HMM(states, observations)
hmm.train(train_text, train_letters)

# Perform recognition using simple Bayes net
simple_result = simple_bayes_recognition(test_letters, train_letters)

# Perform recognition using HMM with Viterbi
hmm_result = hmm.recognize(test_letters)

# Output results
print("Simple: " + simple_result)
print("   HMM: " + hmm_result)

