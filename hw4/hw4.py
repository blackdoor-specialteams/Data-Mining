# coding=utf-8
import csv
import re
#import random
#import math
import TDIDT
import lib
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

#////////////////////////////////////////////////////
def step1(filename):
	print '===========================================\nSTEP 1: TDIDT -- Titanic \n==========================================='
	ddict = dataset_from_file(filename)
	attlist = ["class","age","sex","survived"]
	tree = build_tree_from_dataset(ddict,attlist,"survived")
	kfold = lib.k_folds(ddict,10)
	tdidt_cm = lib.init_nxn(2)

	for f in kfold:
		training,test = lib.holdout_partition(f)
		#s1_tree_run(tdidt_cm,training,test)

	#s6_print_confusion_table("TDIDT", nb)

def s1_tree_run(cm,traning,test):
	key = "survived"
	attlist = ["class","age","sex","survived"]
	tree = build_tree_from_dataset(traning,attlist,"survived")
	pred = []
	actual = []
	for row in test:
		actual.append(row[key]) 
		#pred.append(tree.classify(row,attlist,rules))
	s1_update_cm(cm,actual,pred)

def s1_update_cm(cm,a,p):
	for i in range(len(a)):
		cm[YN_to_int(a[i])][YN_to_int(p[i])] += 1

def YN_to_int(i):
	if i == "yes":
		return 1
	else:
		return 0
def int_to_YN(i):
	if i == 1:
		return "yes"
	else:
		return "no"

def step2(filename):
	return None

def step3(filename):
	return None

def step4(filename):
	return None

#//////////////////////////////////////////////////

def build_tree_from_dataset(dset,attributes,target):
	t = TDIDT.TDIDT(attributes[0],target)
	for row in dset:
		t.put_row(row,attributes)
	print t.view_tree()
	t = t.condense(t)
	#print t
	print t.view_tree()
	return t

def dataset_from_file(filename):
	"""Returns a  list of dictionaries from the file"""
	result = []
	with open(filename, 'r') as f:
		reader = csv.DictReader(f)
		for row in reader:
			result.append(row)
	return result

def main():
	dataset = "titanic.txt"
	step1(dataset)
	
if __name__ == '__main__':
    main()
