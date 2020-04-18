import random
from sklearn.feature_extraction.text import CountVectorizer

# symlink to this file. See https://mg.readthedocs.io/importing-local-python-modules-from-jupyter-notebooks/


def counts_of(corpus, ngram_range):
    vectorizer = CountVectorizer(ngram_range=ngram_range)
    matrix = vectorizer.fit_transform(corpus)
    counts = matrix.toarray().sum(axis=0)
    words = vectorizer.get_feature_names()
    return list(zip(words, counts))


def probabilities_of(word_count):
    words, counts = zip(*word_count)
    total = sum(counts)
    ps = map(lambda x: x / total, counts)
    return list(ps)


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i:i + n]


def randomized_chunks_of(corpus):
    randomized_corpus = random.sample(corpus, len(corpus))
    return list(chunks(randomized_corpus[:2000], 100))


def contains(x, xs):
    if x in xs:
        return True
    else:
        return False


def words_of(word_count):
    return set(map(lambda x: x[0], word_count))


def remove_non_common(word_count1, word_count2):
    words1 = words_of(word_count1)
    words2 = words_of(word_count2)
    wc1 = filter(lambda x: contains(x[0], words2), word_count1)
    wc2 = filter(lambda x: contains(x[0], words1), word_count2)
    return list(wc1), list(wc2)
