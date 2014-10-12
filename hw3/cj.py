# coding=utf-8
import math
import hw2
import csv
import re
import random
import os
import heapq
from tabulate import tabulate

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


def step3(table):
	nb_v1(table)
	return None

def nb_v1():
	training,test = holdout_partition(table)
	learning_nb_v1()

def learning_nb_v1():
	return None
	
#///////////////////////////////////////////////////////
def nb_v2():
	return None

def guassian(x,mean,sdev):
	first, second = 0,0
	if sdev > 0:
		first = 1 / (math.sprt(2*math.pi) * sdev)
		second = math.e ** (-((x - mean) ** 2) / (2 *(sdev ** 2)))
	return first * second

def holdout_partition(table):
	randomized = table [:]
	n = len(table)
	for i in range(n):
		j = randint(0,n-1)
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
	for x in row:
		out += " " + str(x)
	print out

def get_random_indexes(n, size):
	return sorted([int(random.random()*(size-1)) for _ in range(0, n)])

def get_NHTSA_rating(x):
	if x >= 3500:
		return 5
	elif x > 3000:
		return 4
	elif x > 2500:
		return 3
	elif x > 2000:
		return 2 
	else:
		return 1


def get_mpg_rating(x):
	if x >= 45:
		return 10
	elif x >= 37:
		return 9
	elif x >= 31:
		return 8
	elif x >= 27:
		return 7
	elif x >= 24:
		return 6 
	elif x >= 20:
		return 5
	elif x >= 17:
		return 4
	elif x >= 15:
		return 3 
	elif x > 13:
		return 2
	elif x <= 13:
		return 1

def main():
	atts,table = table_from_csv("auto-data-cleaned.txt")
	step3()
	step4()
	step5()

main()