#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import unicode_literals
__author__ = 'DSTC4'
__version__ = "$Revision: 1.0.3 $"

# Common python modules
import os,sys, string
import cPickle as pickle
from lm import ArpaLM
import bleu as bleu

try:
    import numpy as np
except:
    print "Error: Requires numpy from http://www.numpy.org/. Have you installed numpy?"
    sys.exit()

try:
    from sklearn.externals import joblib
    from sklearn.metrics.pairwise import cosine_similarity
except:
    print "Error: Requires sklearn from http://scikit-learn.org/. Have you installed scikit?"
    sys.exit()

# Important directories for the system
root_dir = './scripts/'

# Global variables
NGRAM_ORDER = 3  		            # Order the FM score calculation
FULL_AM_SIZE = 300                 # Size of the trained AM model
OPT_AM_SIZE = 100                  # Optimal value for the trained AM model
PREFIX_AM_FM = 'dstc4'              # Prefix for the AM-FM models

sc = set(['-', "'", '%'])
to_remove = ''.join([c for c in string.punctuation if c not in sc])
table = dict((ord(char), u'') for char in to_remove)

class VSM:
    def __init__(self, model_file, size_am):
        self.am = None
        self.vectorizer = None
        self.load(model_file)
        self.am_components = self.am[:,0:size_am]

    def search(self, ref_sentences, test_sentences):
        """ search for documents that match based on a list of terms """

        assert len(ref_sentences) == len(test_sentences), "ERROR: the length of the reference (%d) and test (%d) " \
                                                          "sentences are not the same" % (len(ref_sentences), len(test_sentences))
        reference_vector = self.vectorizer.transform(ref_sentences)
        target_vector = self.vectorizer.transform(test_sentences)
        cosines = self.cosine_dist(target_vector, reference_vector)
        return cosines

    def cosine_dist(self, target, reference):
        """ related documents j and q are in the concept space by comparing the vectors :
            cosine  = ( V1 * V2 ) / ||V1|| x ||V2|| """
        tgt = np.matrix.dot(target.todense(), self.am_components)
        ref = np.matrix.dot(reference.todense(), self.am_components)
        return max(0, cosine_similarity(ref, tgt)[0])

    def load(self, name_model):
        print('Loading AM model')
        self.am = joblib.load(name_model + '.h5')
        file_h = open(name_model + '.dic', "rb")
        self.vectorizer = pickle.load(file_h)
        file_h.close()

class calcScoresBleuAMFM():
    def __init__(self):

        # Check that the AM models exist
        am_full_matrix = root_dir + '/' + PREFIX_AM_FM + '.' + str(FULL_AM_SIZE)
        if not os.path.isfile(am_full_matrix + '.h5') or not os.path.isfile(am_full_matrix + '.dic'):
            print('******* ERROR: files: ' + am_full_matrix + '.h5 or ' + am_full_matrix + '.dic does not exists.')
            exit(-1)
        elif os.path.getsize(am_full_matrix + '.h5') == 0 or os.path.getsize(am_full_matrix + '.dic') == 0:
            print('******* ERROR: Check if files: ' + am_full_matrix + '.h5 or ' + am_full_matrix + '.dic are not empty.')
            exit(-1)

        # Check that the LM model exists
        lm_model = root_dir + '/' + PREFIX_AM_FM + '.' + str(NGRAM_ORDER) + '.lm'
        if not os.path.exists(lm_model):
            print("******* ERROR: LM file " + lm_model + ' does not exists.')
            exit(-1)
        elif os.path.getsize(lm_model) == 0:
            print("******* ERROR: LM file " + lm_model + ' is empty.')
            exit(-1)

        # Load the models
        self.vs = VSM(am_full_matrix, OPT_AM_SIZE)
        self.lm = ArpaLM(lm_model)

    def doProcessFromStrings(self, ref, pred, id=1, lang='en'):
        ref = self.preProcess(ref)
        pred = self.preProcess(pred)
        return ref, pred

    def preProcess(self, s):
        if len(s) == 0:  # To avoid empty lines
            return '_EMPTY_'

        # Remove some punctuation
        s = s.translate(table)
        # Tokenization
        tokens = s.split()
        new_sent = []
        for token in tokens:
            if token.startswith("%"):
                continue
            if token.endswith("-"):
                token = token[:-1]
            new_sent.append(token)
        s = ' '.join(new_sent).lower()

        return s


    def calculateFMMetric(self, ref, tst):
        sent = '<s> ' + ref.strip() + ' </s>'
        aWords = sent.split()
        num_words_ref = len(aWords) - 2
        prob_ref = 0.0
        # Calculates the log-prob for the different n-grams
        for i in range(1, len(aWords)):
            prob_ref += self.lm.score(tuple(aWords[max(0, i-NGRAM_ORDER+1):i+1]))

        sent = '<s> ' + tst.strip() + ' </s>'
        aWords = sent.split()
        num_words_tst = len(aWords) - 2
        prob_tst = 0.0
        # Calculates the log-prob for the different n-grams
        for i in range(1, len(aWords)):
            prob_tst += self.lm.score(tuple(aWords[max(0, i-NGRAM_ORDER+1):i+1]))

        # Calculates a normalized score for the sentence based on their log-probs and sentences length
        return min(1.0, (prob_ref*num_words_tst)/(prob_tst*num_words_ref))

    def calculateBLEUMetric(self, ref, pred):
        return bleu.calculateBLEU(ref, pred)

    def calculateAMMetric(self, ref, pred):
        return self.vs.search([ref], [pred])
