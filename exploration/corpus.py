from exploration import words as w
from scipy.stats import entropy
import numpy as np


class Corpus:

    def __init__(self, filename):
        with (open(filename)) as f:
            history = f.read()
            self.corpus = history.split('\n')
            self.unigram_count = w.counts_of(self.corpus, (1,1))
            self.bigram_count = w.counts_of(self.corpus, (2,2))
        print('unigram entropy {} in file {}'.format(entropy(w.probabilities_of(self.unigram_count)), filename))
        print('bigram entropy  {} in file {}'.format(entropy(w.probabilities_of(self.bigram_count)), filename))

    def entropy_of(self, ngram_range):
        entropy_results = []
        for chunk in w.randomized_chunks_of(self.corpus):
            h = entropy(w.probabilities_of(w.counts_of(chunk, ngram_range)))
            entropy_results.append(h)

        print('mean = {}, std dev = {} for n-grams in range {}'.format(np.mean(entropy_results), np.std(entropy_results), ngram_range))
        return entropy_results

