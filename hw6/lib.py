import copy
import random
import heapq
import csv
import math
from tabulate import tabulate
from operator import itemgetter

"""
###################################################################################################
###################################################################################################
			General Tool Functions
###################################################################################################
###################################################################################################
	Functions Avaliable:
	-K Fold (dataset, number of folds)
	-Dataset From File (filename)
	-Holdout Partition (dataset)
	-Get Mode (List)
	-Get N random Instances(dataset,N)
	-Get MPG Rating (X)

"""

def k_folds(dataset,k):
	"""
	Returns the statistical mode of xs
	"""
	rdm = copy.deepcopy(dataset) 
	n = len(dataset)
	step = n / k
	for i in range(n):
		j = random.randint(0,n-1)
		rdm[i], rdm[j] = rdm[j],rdm[i]
	return [rdm[i:i + step] for i in range(0, len(rdm), step)]

def get_accuracy(a,p):
	"""
	"""
	count = 0
	for i in range(len(a)):
		if a[i] == p[i]:
			count += 1
	return float(count) / float(len(a))
	
def get_accuracy_and_stdE(a,p):
	"""
	"""
	count = 0
	for i in range(len(a)):
		if a[i] == p[i]:
			count += 1
	acc = float(count) / float(len(a))
	stdE = math.sqrt((acc * (1 - acc)) / len(a))
	return acc ,stdE

def average_acc_and_stdE(xs):
	"""Takes a list of tuples, that average the probablity and standard error of the probablity"""
	p = 0.0
	stdE = 0.0
	for x in xs:
		p += float(x[0])
		stdE += float(x[1])
	p = float(p) / float(len(xs))
	stdE = float(stdE) / float(len(xs))
	return p, stdE

def dataset_from_file(filename):
	"""Returns a  list of dictionaries from the file"""
	result = []
	with open(filename, 'r') as f:
		reader = csv.DictReader(f)
		for row in reader:
			result.append(row)
	return result

def holdout_partition(dataset):
	"""Divides a table into two parts in a 2:1 ratio for length"""
	randomized = dataset [:]
	n = len(dataset)
	for i in range(n):
		j = random.randint(0,n-1)
		randomized[i], randomized[j] = randomized[j],randomized[i]
	n0 = (n*2)/3
	return randomized[0:n0],randomized[n0:]

def get_mode(xs):
	"""
	Returns the statistical mode of xs
	"""
	counts = dict()
	for x in xs:
		if x in counts:
			counts[x] += 1
		else:
			counts[x] = 1
	ret = xs[0]
	for x in counts.keys():
		if counts[x] > counts[ret]:
			ret = x
	return ret

def get_n_rand_instances(dataset,n):
	size = len(dataset)
	index = sorted([int(random.random()*(size-1)) for _ in range(0, n)])
	result = []
	for i in index:
		result.append(dataset[i])
	return result

def get_mpg_rating(x):
	x = float(x)
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

def get_NHTSA_rating(y):
	x = float(y)
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

def guassian(x,mean,sdev):
	"""returns the guassian distance for a value x"""
	first, second = 0,0
	if sdev > 0:
		first = float(1) / float((math.sqrt(2.0*math.pi) * sdev))
		second = math.e ** (-((x - mean) ** 2) / (2.0 *(sdev ** 2)))
	return first * second

def init_nxn(n):
	"""Creates a 2d matrix that is nxn"""
	l =[]
	for i in range(1,n+1):
		l.append([])
		for j in range(1,n+1):
			l[i - 1].append(0)
	return l

def print_confusion_table(name,table):
	"""prints a labeled confusion matrix using tabulate"""
	#table = make_table(file_name,summary_atts)
	print name + " - Stratified 10-Fold Cross Validation"
	headers = ["MPG"] #+ ["Total","Recognition (%)"])
	for i in range(1,11):
		headers.append(str(i))
	headers = headers + ["Total", "Recognition (%)"]
	tmp_table = []
	for i in range(10):
		row = [str(i+1)]
		total = float(sum(table[i]))
		if total > 0:
			rec = float(table[i][i]) / total
		else:
			rec = 0
		for x in table[i]:
			row.append(str(x))
		row.append(str(sum(table[i])))
		row.append(str(rec * 100))
		tmp_table.append(row)
	print tabulate(tmp_table,headers,tablefmt="rst")

def update_cm(cm,a,p):
	for i in range(len(a)):
		if p[i]:
			cm[a[i] -1 ][p[i] - 1] = cm[a[i] -1 ][p[i] - 1] + 1



