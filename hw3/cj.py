# coding=utf-8
import math
import hw2
import csv
import re
import random
import os
import heapq
import numpy
from operator import itemgetter
from tabulate import tabulate
from decimal import Decimal 

#CJ: 3,4,5

'''Create two versions of a Naive Bayes classifier to predict mpg based on the number of cylinders,
weight, and model year attributes. For the first one, create a categorical version of weight using the following
classifications (based on NHTSA vehicle sizes).
Ranking Range
5 >= 3500
4 3000-3499
3 2500-2999
2 2000-2499
1 <= 1999
For the second, calculate the conditional probability for weight using the Gaussian distribution function from
class. Similar to Step 1, test your classifier by selecting random instances from the dataset, predict their
corresponding mpg ranking, and then show their actual mpg ranking:
'''


def step3(table,atts):
	keycol = 1
	checkatts = [2,3,4]
	nb_v1(table,checkatts,keycol)
	nb_v2(table,checkatts,keycol)
	return None

def nb_v1(table,attlist,keycol):
	print "\n" + "Naive Bayes v1"
	nbtable = temp_table_with_NHTSA_rating(table)
	training,test = holdout_partition(nbtable)
	rules = build_all_class_dicts(training,keycol,attlist)
	rand_inst = get_random_indexes(test,5,len(test))

	for row in rand_inst:
		print_instance(row)
		print_classification(row,rules,keycol,attlist)

def print_classification(row,rules,clscol,attlist):
	out = "prediction: " + str(classify(row,attlist,rules)) + ", "
	out += "actual: " + str(row[clscol])
	print out

def classify(inst,attlist,rules):
	prob = []
	for k in rules.keys():
		p = rules[k].get(-1)
		attkeys = rules[k].copy()
		#print attkeys
		attkeys.pop(-1,None) 
		#f
		for j in attkeys.keys():
		#for every attribute in this class's dict
			if inst[j] in rules[k][j].keys():
			#if the value is in the key set
				p = float(p) * float(rules[k][j].get(inst[j]))
			else:
				p = float(0.0)
				break
		prob.append((p,k))
	#sort by prob, then return the first elem, which is the key
	prob.sort(key=lambda x: float(x[0]), reverse=True)
	#print prob
	return prob[0][1]

#////////////////////////////////////////////////////////////////
#BUILD TRAINING DATA STRUCT
#////////////////////////////////////////////////////////////////

def build_all_class_dicts(table,keycol,attlist):
	""" builds a dict of all classes in the dataset, based on the keycol"""
	keydict,total = get_class_keys_and_counts(table,keycol)
	train_dict = {}
	for k in keydict.keys():
		train_dict.update({k:build_class_dict(table,keycol,k,attlist,keydict.get(k),total)})
	return train_dict

def build_class_dict(table,keycol,clskey,attlist,clscount,total):
	"""builds a dict on a single class for the attlist of the table
	Also appends the probablity of this class as -1"""
	cls = {}
	for x in attlist:
		cls.update({x:build_att_dict_for_class(table,keycol,clskey,x,clscount)})
	#add the probablit of the class
	cls.update({-1: (float(clscount) / float(total))})
	return cls


def build_att_dict_for_class(table,keycol,clskey,col,clscount):
	"""Builds the dict of a single attribute for a class on the column"""
	att_d = {}
	#Build dict with counts for each key
	for row in table:
		if row[keycol] == clskey:
			if row[col] not in att_d.keys():
				att_d.update({row[col]:1})
			else:
				att_d[row[col]] = att_d.get(row[col]) + 1
	#get probablity for each key of the attribute
	clsp = {}
	for k in att_d.keys():
		p = (float(att_d.get(k)) / float(clscount))
		clsp.update({k:p})

	return clsp

def get_class_keys_and_counts(table,col):
	"""Finds the count for each class in the table by column, and returns totaa count"""
	cls = {}
	count = 0
	for row in table:
		count += 1
		if  row[col] not in cls:
			cls.update({row[col]:1})
		else:
			cls[row[col]] = cls.get(row[col]) + 1
	return cls, count

#////////////////////////////////////////////////////////////////

def rebuild_table_with_mpg_rating(table):
	for row in table:
		row[1] = get_mpg_rating(row[1])

def temp_table_with_NHTSA_rating(table):
	tmp = table
	for row in tmp:
		row[4] = get_NHTSA_rating(row[4])
	return tmp

def get_NHTSA_rating(y):
	x = float(y)
	if x >= 3500:
		return '5'
	elif x > 3000:
		return '4'
	elif x > 2500:
		return '3'
	elif x > 2000:
		return '2' 
	else:
		return '1'

#///////////////////////////////////////////////////////
def nb_v2(table,attlist,keycol):
	print "\n" + "Naive Bayes v2"
	nbtable = build_table_with_gaussian(table,4)
	training,test = holdout_partition(nbtable)
	rules = build_all_class_dicts(training,keycol,attlist)
	rand_inst = get_random_indexes(test,5,len(test))
	for row in rand_inst:
		print_instance(row)
		print_classification(row,rules,keycol,attlist)

def build_table_with_gaussian(table,col):
	tmp = table
	glist = []
	for row in tmp:
		glist.append(float(row[col]))
	arr = numpy.array(glist)
	mean = numpy.mean(arr,axis=0)
	std = numpy.std(arr,axis=0)
	for row in tmp:
		row[col] = str(guassian(float(row[col]),mean,std))
	return tmp

def guassian(x,mean,sdev):
	first, second = 0,0
	if sdev > 0:
		first = 1 / (math.sqrt(2*math.pi) * sdev)
		second = math.e ** (-((x - mean) ** 2) / (2 *(sdev ** 2)))
	return first * second

def holdout_partition(table):
	randomized = table [:]
	n = len(table)
	for i in range(n):
		j = random.randint(0,n-1)
		randomized[i], randomized[j] = randomized[j],randomized[i]
	n0 = (n*2)/3
	return randomized[0:n0],randomized[n0:]

'''Compute the predictive accuracy (and standard error) of the four classifiers using separate training
and test sets. You should use two approaches for testing. The first approach should use random subsampling
with k = 10. The second approach should use stratified k-fold cross validation with k = 10. Your output
should look something like this (where the ??’s should be replaced by actual values):
'''
def step4():
	return None

'''Create confusion matrices for each classifier. You can use the tabulate package to display your
confusion matrices (it is also okay to format the table manually). Here is an example:
Linear Regression (Stratified 10-Fold Cross Validation):
===== === === === === === === === === === ==== ======= =================
MPG 	1 	2 	3 	4 	5 	6 	7 	8 	9 	10 	Total 	Recognition (%)
===== === === === === === === === === === ==== ======= =================
	1 	14 	2 	5 	3 	1 	0 	0 	3 	0 	0 	25 		56
2 5 3 6 1 1 0 0 0 0 0 16 18.75
3 3 5 9 8 6 0 0 0 0 0 31 29.03
4 0 2 4 18 21 2 3 0 0 0 50 36
5 0 0 0 6 27 15 3 0 0 0 51 52.94
6 0 0 0 1 3 12 15 0 0 0 31 38.71
7 0 0 0 0 1 6 19 0 0 0 26 73.08
8 0 0 0 0 0 1 18 0 1 0 20 0
9 0 0 0 0 0 0 3 0 0 0 3 0
10 0 0 0 0 0 0 0 0 0 0 0 0
===== === === === === === === === === === ==== ======= =================
Naive Bayes I (Stratified 10-Fold Cross Validation):
===== === === === === === === === === === ==== ======= =================
MPG 1 2 3 4 5 6 7 8 9 10 Total Recognition (%)
===== === === === === === === === === === ==== ======= =================
1 20 4 1 0 0 0 0 0 0 0 25 80
2 6 8 2 0 0 0 0 0 0 0 16 50
3 7 6 9 7 2 0 0 0 0 0 31 29.03
4 3 1 7 27 10 2 0 0 0 0 50 54
5 0 0 1 18 22 9 1 0 0 0 51 43.14
6 0 0 0 2 6 17 3 3 0 0 31 54.84
7 0 0 0 0 5 7 11 3 0 0 26 42.31
8 0 0 0 0 1 3 3 13 0 0 20 65
9 0 0 0 0 0 0 0 3 0 0 3 0
10 0 0 0 0 0 0 0 0 0 0 0 0
===== === === === === === === === === === ==== ======= =================
...
'''
def step5():
	return None

''' Use Na¨ıve Bayes and k-nearest neighbor to create two different classifiers to predict survival from the
titanic dataset (titanic.txt). Note that the first line of the dataset lists the name of each attribute (class,
age, sex, and surivived). Your classifiers should use class, age, and sex attributes to determine the survival
class. Be sure to write down any assumptions you make in creating the classifiers. Evaluate the performance
of your classifier using stratified k-fold cross validation (with k = 10) and generate confusion matrices for
the two classifiers.'''

#//////////////////////////////////////////////////////////////////////////////
def table_from_csv(filename):
	"""Returns a table of strings from a CSV file"""
	table =[]
	atts = []
	with open(filename, 'rb') as _in:
		f1 = csv.reader(_in)
		atts = f1.next()
		for row in f1:
			if len(row) > 0:
				table.append(row)
	return atts, table

def print_instance(row):
	out = "instance: "
	for x in range(len(row) - 1):
		out += str(row[x]) + ", "
	out += str(row[len(row) - 1])
	print out

def get_random_indexes(table,n, size):
	index = sorted([int(random.random()*(size-1)) for _ in range(0, n)])
	result = []
	for i in index:
		result.append(table[i])
	return result

def get_mpg_rating(y):
	x = float(y)
	if x >= 45:
		return '10'
	elif x >= 37:
		return '9'
	elif x >= 31:
		return '8'
	elif x >= 27:
		return '7'
	elif x >= 24:
		return '6' 
	elif x >= 20:
		return '5'
	elif x >= 17:
		return '4'
	elif x >= 15:
		return '3' 
	elif x > 13:
		return '2'
	elif x <= 13:
		return '1'

def main():
	atts,table = table_from_csv("auto-data-cleaned.txt")
	rebuild_table_with_mpg_rating(table)
	step3(table,atts)
	#step4()
	#step5()

main()