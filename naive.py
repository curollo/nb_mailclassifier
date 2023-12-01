#!/usr/bin/env python3

import numpy as np

class MNB:
  def __init__(self):
    self.vocab = None
    self.priors = None
    self.likelihoods = None

  def fit(self, X, y):
    # Create the vocabulary
    self.list_arr = X.tolist()
    self.vocab = list(set([word for email in self.list_arr for word in email.split()]))
    self.clean_vocab = [x for x in self.vocab if len(x)<=15 and x.isalpha()]
    #self.vocab_min_two = list(set([x for x in self.clean_vocab if self.clean_vocab.count(x)>=2]))
    self.vocab_final = self.clean_vocab
    self.vocab_size = len(self.vocab_final)

    # Calculate the prior probabilities
    self.priors = {label: (y == label).sum() / len(y) for label in set(y)}

    # Calculate the likelihoods
    self.likelihoods = {
        label: np.ones((self.vocab_size))
        for label in set(y)
    }
    for email, label in zip(X, y):
      for word in email:
        try:
          self.likelihoods[label][self.vocab_final.index(word)] += 1
        except ValueError:
          pass

    # Normalize the likelihoods
    for label in set(y):
      self.likelihoods[label] /= self.likelihoods[label].sum()

  def predict(self, X):
    predictions = []
    for email in X:
      scores = {label: np.log(self.priors[label]) for label in self.priors}
      for word in email:
        if word in self.vocab_final:
          for label in scores:
            try:
              scores[label] += np.log(self.likelihoods[label][self.vocab_final.index(word)])
            except ValueError:
              pass
      predictions.append(max(scores, key=scores.get))
    return predictions
