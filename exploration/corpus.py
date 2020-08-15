from exploration import words as w
from scipy.stats import entropy
import numpy as np
from sklearn.model_selection import cross_val_score


class Corpus:

    def __init__(self, filename):
        with (open(filename)) as f:
            self.lines = f.read()
            self.corpus = self.lines.split('\n')
            self.unigram_count = w.counts_of(self.corpus, (1, 1))
            self.bigram_count = w.counts_of(self.corpus, (2, 2))
        print('unigram entropy {} in file {}'.format(entropy(w.probabilities_of(self.unigram_count)), filename))
        print('bigram entropy  {} in file {}'.format(entropy(w.probabilities_of(self.bigram_count)), filename))

    def entropy_of(self, ngram_range):
        hs = []
        for chunk in w.randomized_chunks_of(self.corpus):
            h = entropy(w.probabilities_of(w.counts_of(chunk, ngram_range)))
            hs.append(h)

        print('mean = {}, std dev = {} for n-grams in range {}'.format(np.mean(hs), np.std(hs), ngram_range))
        return hs

