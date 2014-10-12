# coding=utf-8
from cj import *
import hw2
import csv
import re
import random
import os
import heapq
from tabulate import tabulate

#n: 1,2,6

''' Create a classifier that predicts mpg values using a (least squares) linear regression based on vehicle
weight. Your classifier should take one or more instances, compute the predicted MPG values, and then
map these to the DOE classification/ranking (given in HW 2) for each corresponding instance. Test your
classifier by selecting 5 random instances (via your script) from the dataset, predict their corresponding mpg
ranking, and then show their actual mpg ranking as follows:
===========================================
STEP 1: Linear Regression MPG Classifier
===========================================
instance: 15.0, 8, 400.0, 150.0, 3761, 9.5, 70, 1, chevrolet monte carlo, 3123
class: 4, actual: 3
instance: 17.0, 8, 302.0, 140.0, 3449, 10.5, 70, 1, ford torino, 2778
class: 5, actual: 4
instance: 28.4, 4, 151.0, 90.00, 2670, 16.0, 79, 1, buick skylark limited, 4462
class: 6, actual: 7
instance: 20.0, 6, 232.0, 100.0, 2914, 16.0, 75, 1, amc gremlin, 2798
class: 5, actual: 5
instance: 16.2, 6, 163.0, 133.0, 3410, 15.8, 78, 2, peugeot 604sl, 10990
class: 5, actual: 4
Note you should run your program enough times to check that the approach is working correctly'''
def step1(file_in):
# get lists of weights and mpgs
	weights = []
	mpgs = []
	with open(file_in, 'r') as f:
		reader = csv.DictReader(f)
		for inst in reader:
			weights.append(int(inst['Weight']))
			mpgs.append(float(inst['MPG']))
# get best fit line
	_weight, _mpg, m = hw2.calculate_best_fit_line(weights, mpgs)
# choose random instances from the dataset
	random_indexes = get_random_indexes(5, len(weights))
	random_instances = get_instances(file_in, random_indexes)
# print results of predictions
	print '===========================================\nSTEP 1: Linear Regression MPG Classifier\n==========================================='
	for inst in random_instances:
		print_instance(inst) #print "instance: " + inst['MPG'] + " " + inst['Weight'] + ' ' + inst['Model Year'] + " " + inst['Car Name']
		print 'predicted: ' + str(hw2.get_mpg_rating(get_linear_prediction(_weight, _mpg, m, x = int(inst['Weight'])))) + ' actual: ' + str(hw2.get_mpg_rating(float(inst["MPG"])))

	return None

def get_random_indexes(n, size):
	return sorted([int(random.random()*(size-1)) for _ in range(0, n)])

def get_instances(file_in, indexes):
	random_instances = []
	with open(file_in, 'r') as f:
		reader = csv.DictReader(f)
		i = 0
		for inst in reader:
			if i in indexes:
				random_instances.append(inst)
			i += 1
	return random_instances

def get_linear_prediction(_x, _y, m, x = None, y = None):
	if y == None and x == None:
		return None
	if(y == None): # solve for y
		return m * (x - _x) + _y /1.0
	else:# solve for x
		return (y -_y)/m + _x /1.0

def print_instance(inst, attribs = ["MPG", 'Weight', 'Model Year', 'Car Name']):
	out = "instance:"
	for attrib in attribs:
		out += ' ' + inst[attrib]
	print out


'''Create a k = 5 nearest neighbor classifier for mpg that uses the number of cylinders, weight, and
acceleration attributes to predict mpg. Be sure to normalize the MPG values and also use the Euclidean
distance metric. Similar to Step 1, test your classifier by selecting random instances from the dataset, predict
their corresponding mpg ranking, and then show their actual mpg ranking:
===========================================
STEP 2: k=5 Nearest Neighbor MPG Classifier
===========================================
instance: 15.0, 8, 400.0, 150.0, 3761, 9.5, 70, 1, chevrolet monte carlo, 3123
class: 7, actual: 3
instance: 17.0, 8, 302.0, 140.0, 3449, 10.5, 70, 1, ford torino, 2778
class: 7, actual: 4
instance: 28.4, 4, 151.0, 90.00, 2670, 16.0, 79, 1, buick skylark limited, 4462
class: 1, actual: 7
1instance: 20.0, 6, 232.0, 100.0, 2914, 16.0, 75, 1, amc gremlin, 2798
class: 1, actual: 5
instance: 16.2, 6, 163.0, 133.0, 3410, 15.8, 78, 2, peugeot 604sl, 10990
class: 7, actual: 4
'''
def step2():
	return None

def get_euclidean_d(i, i2):
	return int(math.sqrt(pow((i-i2),2)))

def step6():
	return None


def main():

main()