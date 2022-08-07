# Name: Sashen Moodley
# Student number: 219006946
import time

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Putting the data from the Corpus into a Dataframe
isiZulu_text = pd.read_csv('isiZuluPOSData.csv', delimiter='\t', on_bad_lines='skip')
print(isiZulu_text)

# Getting the subset of the dataframe without the morphological analysis tag
data_for_HMM = isiZulu_text[['TOKEN', 'UPOS']]
print(f"Before removing the trouble rows: {len(data_for_HMM)}")
data_for_HMM = data_for_HMM[data_for_HMM["UPOS"].str.len() > 0]
print(f"After removing the trouble rows: {len(data_for_HMM)}")

# Converting the dataframe subset to list of tuples
tuple_list = list(data_for_HMM.itertuples(index=False, name=None))
print(f"Number of tuples: {len(tuple_list)}")

# Creating sentences structure
sentences = []
start_index = 0

for counter, tuple_element in enumerate(tuple_list):
    if tuple_element[0] == '.':
        sentences.append(list(tuple_list[start_index:counter + 1]))
        start_index = counter + 1

print(f"Number of sentences: {len(sentences)}")

# Splitting data in training and test set
train_set, test_set = train_test_split(sentences, train_size=0.80, test_size=0.20,
                                       random_state=101)  # This will give us the sentences

print(f"Number of training sentences: {len(train_set)}")
print(f"Number of testing sentences: {len(test_set)}")

# Creating list of train and test tagged words
train_tagged_words = [tup for sent in train_set for tup in sent]
test_tagged_words = [tup for sent in test_set for tup in sent]
print(f"Number of tagged training elements: {len(train_tagged_words)}")
print(f"Number of tagged testing elements: {len(test_tagged_words)}")

# Check how many unique tag types
tags = {tag for word, tag in train_tagged_words}
print(f"Number of unique tags: {len(tags)}")
print(tags)

# Check how many unique words
vocabulary = {word for word, tag in test_tagged_words}
print(f"Number of unique words: {len(vocabulary)}")


# Emission Probabilities
def word_given_tag(word, tag, train_bag=train_tagged_words):
    tag_list = [tuple_pair for tuple_pair in train_bag if tuple_pair[1] == tag]
    count_tag = len(tag_list)  # Total number of times the passed tag occurred in the train bag
    w_given_tag_list = [tuple_pair[0] for tuple_pair in tag_list if tuple_pair[0] == word]
    count_w_given_tag = len(w_given_tag_list)  # The total number of times the passed word occurred as the passed tag

    return count_w_given_tag, count_tag


# Transmission Probabilities
def tag2_given_tag1(tag2, tag1, train_bag=train_tagged_words):
    tag_list = [tuple_pair[1] for tuple_pair in train_bag]
    count_tag1 = len([t for t in tag_list if t == tag1])
    count_tag2_tag1 = 0
    for index in range(len(tag_list) - 1):
        if tag_list[index] == tag1 and tag_list[index + 1] == tag2:
            count_tag2_tag1 += 1

    return count_tag2_tag1, count_tag1


# Creating t x t transition matrix of tags, t = no of tags
# Matrix(i, j) represents P(jth tag after the ith tag)
tags_matrix = np.zeros((len(tags), len(tags)), dtype='float32')
for i, tag1 in enumerate(list(tags)):
    for j, tag2 in enumerate(list(tags)):
        tags_matrix[i, j] = tag2_given_tag1(tag2=tag2, tag1=tag1)[0] / tag2_given_tag1(tag2=tag2, tag1=tag1)[1]

tags_df = pd.DataFrame(tags_matrix, columns=list(tags), index=list(tags))


# The manually-implemented viterbi algorithm used for decoding
def viterbi(words, train_bag=train_tagged_words):
    state = []
    tag_labels = list({tuple_pair[1] for tuple_pair in train_bag})

    for key, word in enumerate(words):
        # initialize list of probabilities column for a given observation
        state_probabilities = []
        for tag in tag_labels:
            if key == 0:
                transition_p = tags_df.loc['PUNC', tag]  # First word in the sentence
            else:
                transition_p = tags_df.loc[state[-1], tag]

            # calculating the emission and state probabilities
            emission_p = word_given_tag(words[key], tag)[0] / word_given_tag(words[key], tag)[1]
            state_probability = emission_p * transition_p
            state_probabilities.append(state_probability)

        max_state_prob = max(state_probabilities)  # maximum state probability
        # getting state for which probability is maximum
        state_max = tag_labels[state_probabilities.index(max_state_prob)]
        state.append(state_max)
    return list(zip(words, state))  # creates a mapping


# list of x sentences on which we test the model
test_run = test_set[:20]  # -> this will change accordingly
print(f"Number of testing sentences being used: {len(test_run)}")

# list of words with their tags
test_run_base = [tup for sent in test_run for tup in sent]

# list of words without their tags
test_tagged_words = [tup[0] for sent in test_run for tup in sent]
print(f"Size of test_tagged_words: {len(test_tagged_words)}")

print("Starting testing...")
start = time.time()
tagged_seq = viterbi(test_tagged_words)  # our training tagged words is the default keyword argument
end = time.time()
print("Test end....")

print(f"Time taken: {end - start} seconds")

# accuracy
check = [i for i, j in zip(tagged_seq, test_run_base) if i == j]

accuracy = len(check) / len(tagged_seq)
print(f"Veterbi Algorithm Accuracy: ", accuracy * 100)
