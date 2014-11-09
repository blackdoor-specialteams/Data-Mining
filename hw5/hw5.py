# coding=utf-8

import re
import math
import ENSEMBLE
import TDIDT
import lib
import operator
from tabulate import tabulate


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
	print '===========================================\nSTEP 1: ENSEMBLE -- ' + filename + ' \n========================================='
	ds = lib.dataset_from_file(filename)
	k=3
	folds = lib.k_folds(ds,k)

	en = []
	std = []
	for i in range(k):
		training = []
		for f in range(k):
			if f != i:
				training = training + folds[f]
		test = folds[i]
		en.append(s1_ensemble(training,test,filename))
		std.append(s1_standard(training,test))
	s1_print_results(en,std)

def s1_ensemble(training,test,filename):
	target = "class"
	N = 150
	M = 60
	F = 0
	if filename == "agaricus-lepiota.txt":
		F = 6
	elif filename == "tic-tac-toe.txt":
		F = 4

	Eclass = ENSEMBLE.ENSEMBLE(target,N,M,F)
	Eclass.grow_forest(training)
	actual = []
	pred = []
	for x in test:
		actual.append(x[target])
		pred.append(Eclass.trackrecord_classify(x))
	return lib.get_accuracy_and_stdE(actual,pred)

def s1_standard(training,test):
	target = "class"
	t = TDIDT.TDIDT(target)
	t.put_dataset(training,training[0].keys())
	t.condense(t)

	actual = []
	pred = []
	for x in test:
		actual.append(x[target])
		pred.append(t.classify(x))
	return lib.get_accuracy_and_stdE(actual,pred)

def s1_print_results(ens,std):
	a,e = lib.average_acc_and_stdE(ens)
	print "ENSEMBLE\nAccuracy: " + "{:.5f}".format(a) + " +- " + "{:.5f}".format(e)
	a,e = lib.average_acc_and_stdE(std)
	print "Standard Tree\nAccuracy: " + "{:.5f}".format(a) + " +- " + "{:.5f}".format(e)
	return None

#/////////////////////////////////////////////////////////////////	
def step2(filename):
	print '===========================================\nSTEP 2: ENSEMBLE -- ' + filename + ' \n========================================='
	ds = lib.dataset_from_file(filename)
	k=3
	folds = lib.k_folds(ds,k)
	test = None
	if filename == "agaricus-lepiota.txt":
		test = [(5,3,4),(10,5,7),(20,10,10),(60,20,13),(100,5,3),(100,50,5),(100,30,4)]
	elif filename == "tic-tac-toe.txt":
		test = [(5,3,4),(20,7,7),(60,30,9),(100,15,4),(150,10,6),(150,30,3),(100,10,3)]
	for x in test:
		en = []
		std = []
		for i in range(k):
			training = []
			for f in range(k):
				if f != i:
					training = training + folds[f]
			test = folds[i]
			en.append(s2_ensemble(training,test,x))
		s2_print_results(en,x)

def s2_ensemble(training,test,para):
	target = "class"
	N = para[0]
	M = para[1]
	F = para[2]

	Eclass = ENSEMBLE.ENSEMBLE(target,N,M,F)
	Eclass.grow_forest(training)
	actual = []
	pred = []
	for x in test:
		actual.append(x[target])
		pred.append(Eclass.trackrecord_classify(x))
	return lib.get_accuracy_and_stdE(actual,pred)

def s2_print_results(ens,para):
	print "Parameters \nN: " + str(para[0]) + " M: " + str(para[1]) + " F: " + str(para[2])
	a,e = lib.average_acc_and_stdE(ens)
	print "ENSEMBLE\nAccuracy: " + "{:.5f}".format(a) + " +- " + "{:.5f}".format(e) + "\n"

#////////////////////////////////////////////////////////////////
def main():
	mushroom = "agaricus-lepiota.txt"
	tictactoe = "tic-tac-toe.txt"

	step1(mushroom)
	step2(mushroom)
	
	step1(tictactoe)
	step2(tictactoe)

if __name__ == '__main__':
    main()
