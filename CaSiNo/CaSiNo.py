#https://convokit.cornell.edu/documentation/casino-corpus.html#:~:text=Data%20License-,CaSiNo%20Corpus,Association%20for%20Computational%20Linguistics.

from convokit import Corpus, download
corpus = Corpus(filename=download("casino-corpus"))
corpus.dump("casino")