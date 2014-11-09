# coding=utf-8

import re
import math
import ENSEMBLE
import TDIDT
import lib
import operator
from tabulate import tabulate

def build_track_records(classifiers, dataset, attributes, target):
	"""

	"""
	track_records = []
	for classifier in classifiers:
		record = dict()
		for entry in dataset:
			prediction = classifier.predict(target, attributes)
			if prediction in record:
				if entry[target] in prediction:
					entry[target] += 1
				else:
					entry[target] = 1
			else:
				record[prediction][entry[target]] = 1
		track_records.append(record)
	return track_records

def get_record_prediction(classifiers, target, records, attributes):
	votes = dict()
	for classifier in range(0,len(classifiers)):
		prediction = classifiers[classifier].predict(target, attributes)
		for v in records[classifiers[classifier]][prediction].keys():
			if v in votes:
				votes[v] += records[classifiers[classifier]][prediction][v]
			else:
				votes[v] = records[classifiers[classifier]][prediction][v]
	winner = None
	for vote in votes.keys():
		if winner == None or votes[vote] > votes[winner]:
			winner = vote
	return winner

'''
################################################################################
################################################################################
People's Republic of North Lead By Glorious Party Leader Nate
DMZ 214
Southern Provience of Code, True Heir to the Throne: CJ
################################################################################
################################################################################
'''
def step1(filename):
	print '===========================================\nSTEP 1: ENSEMBLE -- MUSHROOM \n========================================='
	ds = lib.dataset_from_file(filename)
	k=3
	folds = lib.k_folds(ds,k)
	target = "Class-Label"

	en = []
	std = []
	for i in range(k):
		training = []
		for f in range(k):
			if f != i:
				training = training + folds[f]
		test = folds[i]
		en.append(s1_ensemble(training,test))
		std.append(s1_standard(training,test))
	s1_print_results(en,std)

def s1_ensemble(training,test):
	target = "Class-Label"
	N = 5
	M = 3
	F = 4
	Eclass = ENSEMBLE.ENSEMBLE(target,N,M,F)
	Eclass.grow_forest(training)
	actual = []
	pred = []
	for x in test:
		actual.append(x[target])
		pred.append(Eclass.classify(x))
	return lib.get_accuracy_and_stdE(actual,pred)

def s1_standard(training,test):
	target = "Class-Label"
	t = TDIDT.TDIDT(target)
	t.put_dataset(training,wrap)

	actual = []
	pred = []
	for x in test:
		actual.append(x[target])
		pred.append(t.classify(x))
	return lib.get_accuracy_and_stdE(actual,pred)

def s1_print_results(ens,std):
	target = "Class-Label"
	a,e = lib.average_acc_and_stdE(ens)
	print "ENSEMBLE\nAccuracy: " + str(a) + " +- " + str(e)
	a,e = lib.average_acc_and_stdE(std)
	print "Standard Tree\nAccuracy: " + str(a) + " +- " + str(e)
	return None



#/////////////////////////////////////////////////////////////////	
def step2(filename):
	print '===========================================\nSTEP 2: TDIDT -- Auto Data \n========================================='
	return None

def main():
	titanic = "titanic.txt"
	autodata = "auto-data-cleaned.txt"
	mushroom = "agaricus-lepiota.txt"
	tictactoe = "tic-tac-toe.txt"

	step1(mushroom)
	
if __name__ == '__main__':
    main()
