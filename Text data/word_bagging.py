#                                   Representing Text Data as a Bag of Words

# One of the most simple but effective and commonly used ways to represent text for machine learning is using the
# bag-of-words representation. When using this representation, we discard most of the structure of the input text, like
# chapters, paragraphs, sentences, and formatting, and only count how often each word appears in each text in the
# corpus. Discarding the structure and counting only word occurrences leads to the mental image of representing text
# as a “bag.”

import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.naive_bayes import MultinomialNB

df = pd.read_csv("spam.csv")

# Setting the target and the data, and splitting into train and test

df["spamm"] = df["Category"].apply(lambda x: 1 if x == "spam" else 0)

target = df.spamm
data = df.Message

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=42)

# Fitting and transforming the CountVectorizer consists of the tokenization of the training data and building of the
# vocabulary, which can be accessed as the vocabulary_ attribute

v = CountVectorizer(min_df=5)
X_train_token = v.fit_transform(X_train)

# We can set the minimum number of documents a token needs to appear in with the min_df parameter, this helps to reduce
# the dimensions of the dataset

print("Vocabulary size: {}".format(len(v.vocabulary_)))
print("Vocabulary content:\n {}".format(v.vocabulary_))

# To create the bag-of-words representation for the training data, we call the transform method
print("Dense representation of bag_of_words:\n{}".format(X_train_token.toarray()))

param_grid = {'alpha': [0.001, 0.01, 0.1, 1, 10]}
grid = GridSearchCV(MultinomialNB(), param_grid, cv=5)
grid.fit(X_train_token, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters: ", grid.best_params_)

emails = [
    'Hey mohan, can we get together to watch football game tomorrow?',
    'Upto 20% discount on parking, exclusive offer just for you. Dont miss this reward!'
]
emails_count = v.transform(emails)
predicted_emails = grid.predict(emails_count)
print(predicted_emails)

print()
print()

#                                                   Stopwords

# Another way that we can get rid of uninformative words is by discarding words that are too frequent to be informative.
# There are two main approaches: using a languagespecific list of stopwords, or discarding words that appear too
# frequently. scikitlearn has a built-in list of English stopwords in the feature_extraction.text module

from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS

print("Number of stop words: {}".format(len(ENGLISH_STOP_WORDS)))
print("Every 10th stopword:\n{}".format(list(ENGLISH_STOP_WORDS)[::10]))

vect = CountVectorizer(min_df=5, stop_words="english")
X_train_token2 = vect.fit_transform(X_train)

grid2 = GridSearchCV(MultinomialNB(), param_grid, cv=5)
grid2.fit(X_train_token2, y_train)
print("Best cross-validation score: {:.2f}".format(grid2.best_score_))
print("Best parameters: ", grid2.best_params_)

# In this case there was a reduction of 198 characteristics, and the performance of the model stayed the same. The main
# advantage is the increase in the used time to complete the task.

print()
print()

#                           Bag-of-Words with More Than One Word (n-Grams)

# One of the main disadvantages of using a bag-of-words representation is that word order is completely discarded.
# Therefore, the two strings “it’s bad, not good at all” and “it’s good, not bad at all” have exactly the same
# representation, even though the meanings are inverted. Putting “not” in front of a word is only one example (if an
# extreme one) of how context matters. Fortunately, there is a way of capturing context when using a bag-of-words
# representation, by not only considering the counts of single tokens, but also the counts of pairs or triplets of
# tokens that appear next to each other Pairs of tokens are known as bigrams, triplets of tokens are known as trigrams,
# and more generally sequences of tokens are known as n-grams. The ngram_range parameter is a tuple, consisting of the
# minimum length and the maximum length of the sequences of tokens that are considered

from sklearn.pipeline import make_pipeline

pipe = make_pipeline(CountVectorizer(min_df=5), MultinomialNB())
# running the grid search takes a long time because of the relatively large grid and the inclusion of trigrams


param_grid = {"multinomialnb__alpha": [0.001, 0.01, 0.1, 1, 10, 100],
              "countvectorizer__ngram_range": [(1, 1), (1, 2), (1, 3)]}
grid = GridSearchCV(pipe, param_grid, cv=5)
grid.fit(X_train, y_train)
print("Best cross-validation score: {:.2f}".format(grid.best_score_))
print("Best parameters:\n{}".format(grid.best_params_))




