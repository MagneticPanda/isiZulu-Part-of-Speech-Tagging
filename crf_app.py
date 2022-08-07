# Name: Sashen Moodley
# Student number: 219006946
import re
import string
import time

import pandas as pd
import sklearn_crfsuite
from sklearn_crfsuite import metrics
from sklearn_crfsuite.metrics import _flattens_y
from collections import Counter


@_flattens_y
def flat_classification_report(y_true, y_pred, labels=None, **kwargs):  # Fixing the keyword argument bug
    """
    Return classification report for sequence items.
    """
    from sklearn import metrics
    return metrics.classification_report(y_true, y_pred, labels=labels, **kwargs)


data = {}

isiZulu_text = pd.read_csv('isiZuluPOSData.csv', sep='\t', on_bad_lines='skip')
print(isiZulu_text)

data_without_morph = isiZulu_text[['TOKEN', 'UPOS']]  # Dropping the morphological segmentation column
data_without_morph = data_without_morph[data_without_morph["UPOS"].str.len() > 0]  # Removing rows with blank Tags

# Doing a 80:20 split between the training and testing data
training_set = data_without_morph.iloc[:39286]
training_tuple = list(training_set.itertuples(index=False, name=None))
testing_set = data_without_morph.iloc[39286:]
testing_tuple = list(testing_set.itertuples(index=False, name=None))

training_sentences = []
start_index = 0

for counter, tuple_element in enumerate(training_tuple):
    if tuple_element[0] == '.':
        training_sentences.append(list(training_tuple[start_index:counter + 1]))
        start_index = counter + 1

testing_sentences = []
start_index = 0

for counter, tuple_element in enumerate(testing_tuple):
    if tuple_element[0] == '.':
        testing_sentences.append(list(testing_tuple[start_index:counter + 1]))
        start_index = counter + 1

data['train_sent'] = training_sentences
data['test_sent'] = testing_sentences


# Converting our word to feature set
def word_to_features(sent, i):
    """
    :param sent: This is the sentence we pass through
    :param i: This is the index of the word-tag pair in the sentence
    :return: A dictionary which breaks down the word into a set of features
    """

    word = sent[i][0]  # Extracting the word from the sentence
    post_tag = sent[i][1]  # Extracting the tag for the word

    # Several feature formats are supported, here I am using 'feature dicts'
    # Our initial set of features whose values will be modified as we move along
    features = {  # these were the features I found from looking at examples
        'bias': 1.0,
        'word': word,
        'len(word)': len(word),
        'word[:4]': word[:4],
        'word[:3]': word[:3],
        'word[:2]': word[:2],
        'word[-3:]': word[-3:],
        'word[-2:]': word[-2:],
        'word[-4:]': word[-4:],
        'word.lower()': word.lower(),
        'word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)', r'\1', word.lower()),
        'word.ispunctuation': (word in string.punctuation),
        'word.isdigit()': word.isdigit()
    }

    if i > 0:  # Not the first word in the sentence
        word1 = sent[i - 1][0]  # Getting the preceding word in the sentence
        # Adding features about the previous word the feature set of this word
        features.update({
            '-1:word': word1,
            '-1:len(word)': len(word1),
            '-1:word.lower()': word1.lower(),
            '-1:word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)', r'\1', word1.lower()),
            '-1:word[:3]': word1[:3],
            '-1:word[:2]': word1[:2],
            '-1:word[-3:]': word1[-3:],
            '-1:word[-2:]': word1[-2:],
            '-1:word.isdigit()': word1.isdigit(),
            '-1:word.ispunctuation': (word1 in string.punctuation)
        })
    else:
        features['BOS'] = True  # Beginning of sentence

    if i > 1:
        word2 = sent[i - 2][0]  # Getting word 2 words back in sentence history
        # Adding features about the 2nd preceding word
        features.update({
            '-2:word': word2,
            '-2:len(word)': len(word2),
            '-2:word.lower()': word2.lower(),
            '-2:word[:3]': word2[:3],
            '-2:word[:2]': word2[:2],
            '-2:word[-3:]': word2[-3:],
            '-2:word[-2:]': word2[-2:],
            '-2:word.isdigit()': word2.isdigit(),
            '-2:word.ispunctuation': (word2 in string.punctuation)
        })

    if i < len(sent) - 1:
        word1 = sent[i + 1][0]  # Getting the next word in the sentence
        # Adding features about the next word
        features.update({
            '+1:word': word1,
            '+1:len(word)': len(word1),
            '+1:word.lower()': word1.lower(),
            '+1:word[:3]': word1[:3],
            '+1:word[:2]': word1[:2],
            '+1:word[-3:]': word1[-3:],
            '+1:word[-2:]': word1[-2:],
            '+1:word.isdigit()': word1.isdigit(),
            '+1:word.ispunctuation': (word1 in string.punctuation)
        })
    else:
        features['EOS'] = True  # End of sentence

    if i < len(sent) - 2:
        word2 = sent[i + 2][0]  # Getting 2nd following word
        # Adding features about 2nd following word
        features.update({
            '+2:word': word2,
            '+2:len(word)': len(word2),
            '+2:word.lower()': word2.lower(),
            '+2:word.stemmed': re.sub(r'(.{2,}?)([aeiougyn]+$)', r'\1', word2.lower()),
            '+2:word[:3]': word2[:3],
            '+2:word[:2]': word2[:2],
            '+2:word[-3:]': word2[-3:],
            '+2:word[-2:]': word2[-2:],
            '+2:word.isdigit()': word2.isdigit(),
            '+2:word.ispunctuation': (word2 in string.punctuation)
        })

    return features

# Converting sentence to features
def sent_to_features(sent):
    """
    :param sent: takes in a sentence which is represented as a list of word-tag pairings
    :return: list of dictionaries for each word in the sentence
    """
    return [word_to_features(sent, i) for i in range(len(sent))]

# Converting sentence to labels
def sent_to_labels(sent):
    """
    :param sent: takes in a sentence which is represented as a list of word-tag pairings
    :return: list of all the tags assigned to each word in the sentence (in order of the words)
    """
    return [word[1] for word in sent]

# Converting sentence to tokens
def sent_to_tokens(sent):
    """
    :param sent: takes in a sentence which is represented as a list of word-tag pairings
    :return: list of all the words in the sentence without their tags (in order)
    """
    return [word[0] for word in sent]



X_train = [sent_to_features(s) for s in training_sentences]
print(f"XTrain size: {len(X_train)}")
y_train = [sent_to_labels(s) for s in training_sentences]
print(f"yTrain size: {len(y_train)}")

X_test = [sent_to_features(s) for s in testing_sentences]
print(f"XTest size: {len(X_test)}")
y_test = [sent_to_labels(s) for s in testing_sentences]
print(f"y_test size: {len(y_test)}")

# Training the Model
crf = sklearn_crfsuite.CRF(
    algorithm='lbfgs',  # L-BFGS training algorithm with Elastic Net (L1 + L2) regularization
    c1=0.25,
    c2=0.3,
    max_iterations=100,
    all_possible_transitions=True
)
print("Fitting data into model....")
start_time = time.time()
crf.fit(
    X=X_train,
    y=y_train,
    X_dev=None,
    y_dev=None
)
end_time = time.time()
print(f"Model fitting completed in: {end_time-start_time}")

# Obtaining testing metrics
labels = list(crf.classes_)
print(labels)

print("Testing prediction being made...")
start_time = time.time()
y_pred = crf.predict(X_test)
end_time = time.time()
print(f"Testing prediction completed in: {end_time - start_time}")

print(f"y_pred size: {len(y_pred)}")
print(f"F1 score on the test set: {metrics.flat_f1_score(y_test, y_pred, average='weighted', labels=labels)}")
print(f"Accuracy on the test set: {metrics.flat_accuracy_score(y_test, y_pred)}")

# Inspecting per class (label) results
sorted_labels = sorted(
    labels,
    key=lambda name: (name[1:], name[0])
)
print("Test set classification report: ")
print(flat_classification_report(y_test, y_pred, labels=sorted_labels, digits=3))
print()


# Obtaining training metrics
print("Training Prediction being made...")
start_time = time.time()
y_pred = crf.predict(X_train)
end_time = time.time()
print(f"Training prediction completed in: {end_time - start_time}")

print(f"y_pred size: {len(y_pred)}")
print(f"F1 score on the train set: {metrics.flat_f1_score(y_train, y_pred, average='weighted', labels=labels)}")
print(f"Accuracy on the train set: {metrics.flat_accuracy_score(y_train, y_pred)}")


print("Train set classification report: ")
print(flat_classification_report(y_train, y_pred, labels=sorted_labels, digits=3))
print()


def print_transitions(transition_features):
    for (label_from, label_to), weight in transition_features:
        print("%-6s -> %-7s %0.6f" % (label_from, label_to, weight))


print("Top 10 likely transition - \n")
print_transitions(Counter(crf.transition_features_).most_common(10))

print("\nTop 10 unlikely transitions - \n")
print_transitions(Counter(crf.transition_features_).most_common()[-10:])
