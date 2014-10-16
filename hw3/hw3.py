# coding=utf-8
import hw2
import csv
import re
import random
import math
import os
from operator import itemgetter
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
def step2(file_in):
	print "===========================================\nSTEP 2: k=5 Nearest Neighbor MPG Classifier\n==========================================="
	random_indexes = get_random_indexes(5, 315)
	random_instances = get_instances(file_in, random_indexes)
	training_set = []
	with open(file_in, 'r') as f:
		reader = csv.DictReader(f)
		for inst in reader:
			training_set.append(inst)
	for inst in random_instances:
		print_instance(inst, attribs = ["MPG", "Cylinders", "Weight", "Acceleration", 'Model Year', 'Car Name']) #print "instance: " + inst['MPG'] + " " + inst['Weight'] + ' ' + inst['Model Year'] + " " + inst['Car Name']
		print 'predicted: ' + str(get_mpg_class_label(get_knn(inst, 5, training_set))) + ' actual: ' + str(hw2.get_mpg_rating(float(inst["MPG"])))

	return None

def get_mpg_class_label(knn):
	labels = []
	for neighbor in knn:
		labels.append(hw2.get_mpg_rating(float(neighbor["MPG"])))
	return get_mode(labels)


def get_knn(instance, k, training_set, attribs = ["Cylinders", "Weight", "Acceleration"]):
	'''
	Returns the k nearest neighbors to instance based on training_file and attribs
	'''
	ranges = get_ranges(training_set, attribs = attribs)
	distances = []
	for inst in training_set:
		distances.append( [get_neighbor_d(instance, inst, ranges, attribs), inst] )
	distances = sorted(distances, key=itemgetter(0))
	ret = [e[1] for e in distances[0:k]]
	return ret

def get_neighbor_d(instance, neighbor, ranges, attribs = ["Cylinders", "Weight", "Acceleration"]):
	"""
	Returns the distance between instance and neighbor.
	The distance between two instances is the sum of the Euclidean distances between the normalized values of each attribute in attribs. Non-Ordered attributes have a distance 1 if their values are equal, else 0.
	"""
	d = 0
	for attrib in attribs:
		if float(ranges[attrib]) == 0:
			print ranges
		else:
			try:
				d += get_euclidean_d(float(instance[attrib])/float(ranges[attrib]), float(neighbor[attrib])/float(ranges[attrib]))
			except ValueError:
				if instance[attrib] == neighbor[attrib]:
					d += 1
	return d


def get_ranges(data_set, attribs = ["Cylinders", "Weight", "Acceleration"]):
	"""
	Returns a dictionary which has attribute labels as keys and the range of the attributes under that label in data_set as keys
	"""
	ranges = dict()
	for attrib in attribs:
		xs = []
		for inst in data_set:
			xs.append(float(inst[attrib]))
		ranges[attrib] = max(xs) - min(xs)
	return ranges

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

def get_euclidean_d(i, i2):
	"""
	Returns the Euclidean distance between i and i2
	"""
	return float(math.sqrt(pow((i-i2),2)))

def table_to_lick_dicts(table, attribs = ["Acceleration","MPG","Model Year","Cylinders","Weight","Displacement","Car Name","Horsepower","Origin","MSRP"]):
	list_dicts = []
	for row in table:
		entry = dict()
		i = 0
		for attrib in attribs:
			entry[attrib] = row[i]
			i += 1
		list_dicts.append(entry)
	return list_dicts

'''Use Na¨ıve Bayes and k-nearest neighbor to create two different classifiers to predict survival from the
titanic dataset (titanic.txt). Note that the first line of the dataset lists the name of each attribute (class,
age, sex, and surivived). Your classifiers should use class, age, and sex attributes to determine the survival
class. Be sure to write down any assumptions you make in creating the classifiers. Evaluate the performance
of your classifier using stratified k-fold cross validation (with k = 10) and generate confusion matrices for
the two classifiers.'''
def step6():
	return None


'''
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
People's Republic of North Lead By Glorious Party Leader Nate
DMZ 214
CJ'z code
################################################################################
################################################################################
################################################################################
################################################################################
################################################################################
'''
import math
import csv
import random
from copy import deepcopy
from tabulate import tabulate
#CJ: 4,5

def step3(table,atts):
	print '===========================================\nSTEP 3: Naive Bayes MPG Classifiers\n==========================================='
	keycol = 1
	checkatts = [2,3,4]
	print "Naive Bayes I"
	nb_v1(deepcopy(table),checkatts,keycol)
	print "Naive Bayes v2"
	nb_v2(deepcopy(table),checkatts,keycol)

def nb_v1(table,attlist,keycol):
	nbtable = temp_table_with_NHTSA_rating(table)
	rules = build_all_class_dicts(nbtable,keycol,attlist)
	rand_inst = get_random_table_indexes(nbtable,5,len(nbtable))
	for row in rand_inst:
		print_row(row)
		print_classification(row,rules,keycol,attlist)

def nb_v2(table,attlist,keycol):
	nbtable = build_table_with_gaussian(table,4)
	rules = build_all_class_dicts(nbtable,keycol,attlist)
	rand_inst = get_random_table_indexes(nbtable,5,len(nbtable))
	for row in rand_inst:
		print_row(row)
		print_classification(row,rules,keycol,attlist)

def print_classification(row,rules,clscol,attlist):
	out = "prediction: " + nb_classify(row,attlist,rules) + ", "
	out += "actual: " + row[clscol]
	print out

def nb_classify(inst,attlist,rules):
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

def rebuild_table_with_mpg_rating(table):
	for row in table:
		row[1] = str(hw2.get_mpg_rating(float(row[1])))
		tuple(row)

def temp_table_with_NHTSA_rating(table):
	tmp = table[:]
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

def build_table_with_gaussian(table,col):
	tmp = table[:]
	glist = [float(tmp[i][4]) for i in xrange(len(tmp))]
	mean = float(sum(glist)) / float(len(glist))
	std = math.sqrt((float(sum(glist) ** 2) / float(len(glist))) - mean)
	for row in tmp:
		row[col] = str(guassian(float(row[col]),mean,std))
	return tmp

def guassian(x,mean,sdev):
	first, second = 0,0
	if sdev > 0:
		first = float(1) / float((math.sqrt(2.0*math.pi) * sdev))
		second = math.e ** (-((x - mean) ** 2) / (2.0 *(sdev ** 2)))
	return first * second

def holdout_partition(table):
	randomized = table [:]
	n = len(table)
	for i in range(n):
		j = random.randint(0,n-1)
		randomized[i], randomized[j] = randomized[j],randomized[i]
	n0 = (n*2)/3
	return randomized[0:n0],randomized[n0:]

def step4(table,atts):
	print '===========================================\nSTEP 4: Predictive Accuracy\n==========================================='
	first_approach(deepcopy(table))
	second_approach(deepcopy(table))

def first_approach(table):
	"""Random Subsampling"""
	print "Random Subsample (k=10, 2:1 Train/Test)"

	nb_v1 = []
	nb_v2 = []
	knn = []
	lnr = []
	
	for x in range(0,10):
		training,test = holdout_partition(table)
		#nb_v1.append(s4_NB_v1(deepcopy(training),deepcopy(test)))
		#nb_v2.append(s4_NB_v2(deepcopy(training),deepcopy(test)))
		lnr.append(s4_LR(deepcopy(training),deepcopy(test)))
		knn.append(s4_KNN(deepcopy(training),deepcopy(test)))

	print_predAcc_format("Linear Reression",lnr)
	print_predAcc_format("K-NN ",knn)
	#print_predAcc_format("Naive Bayes I",nb_v1)
	#print_predAcc_format("Naive Bayes II",nb_v2)

def second_approach(table):
	"""K-fold cross validation, k == 0"""
	print "Stratified 10-Fold Cross Validation"
	nb_v1 = []
	nb_v2 = []
	knn = []
	lnr = []
	kfold = k_folds(table,10)
	for f in kfold:
		training,test = holdout_partition(f)

		nb_v1.append(s4_NB_v1(deepcopy(training),deepcopy(test)))
		nb_v2.append(s4_NB_v2(deepcopy(training),deepcopy(test)))
		lnr.append(s4_LR(deepcopy(training),deepcopy(test)))
		knn.append(s4_KNN(deepcopy(training),deepcopy(test)))

	print_predAcc_format("Linear Reression",lnr)
	print_predAcc_format("K-NN ",knn)
	print_predAcc_format("Naive Bayes I",nb_v1)
	print_predAcc_format("Naive Bayes II",nb_v2)

def k_folds(table,k):
	rdm = deepcopy(table) 
	n = len(table)
	for i in range(n):
		j = random.randint(0,n-1)
		rdm[i], rdm[j] = rdm[j],rdm[i]
	return [rdm[i:i + k] for i in range(0, len(rdm), k)]

def s4_KNN(training, test):
	nnn = 5
	training = table_to_lick_dicts(training)
	test = table_to_lick_dicts(test)
	#for instance in test:
	#	actuals.append(hw2.get_mpg_rating(float(test["MPG"])))
	#	predictions.append(get_mpg_class_label(get_knn(instance, k, training)))
	clset = {}
	for row in test:
		#print row["MPG"]
		k = str(hw2.get_mpg_rating(row.get("MPG")))
		prd = str(get_mpg_class_label(get_knn(row, nnn, training)))
		if k not in clset:
			clset[k] = run_single_test(k,prd,0,0)
		else:
			tmp = clset.get(k)
			clset[k] = run_single_test(k,prd,tmp[0],tmp[1])
	p = 0.0
	for k in clset.keys():
		tmp = clset.get(k)
		p += calculate_p(tmp[0],tmp[1])
	p = float(p) / float(10)
	se = calculate_stdE(p,len(test))
	return p, se

def s4_LR(training,test):
	keycol = 1
	weights = []
	mpgs = []
	for row in training:
		weights.append(float(row[4]))
		mpgs.append(float(row[1]))
	_weight, _mpg, m = hw2.calculate_best_fit_line(weights, mpgs)
	clset = {}
	for row in test:
		k = str(hw2.get_mpg_rating(row[keycol]))
		prd = str(hw2.get_mpg_rating(get_linear_prediction(_weight, _mpg, m, x = float(row[4]))))
		if k not in clset:
			clset[k] = run_single_test(k,prd,0,0)
		else:
			tmp = clset.get(k)
			clset[k] = run_single_test(k,prd,tmp[0],tmp[1])
	p = 0.0
	for k in clset.keys():
		tmp = clset.get(k)
		p += calculate_p(tmp[0],tmp[1])
	p = float(p) / float(10)
	se = calculate_stdE(p,len(test))
	return p, se

def s4_NB_v1(train,test):
	nbtable = temp_table_with_NHTSA_rating(train)
	return run_NB(nbtable,test)

def s4_NB_v2(train,test):
	nbtable = build_table_with_gaussian(train,4)
	return run_NB(nbtable,test)

def run_NB(nbtable,test):
	keycol = 1
	attlist = [2,3,4]
	rules = build_all_class_dicts(nbtable,keycol,attlist)
	rand_inst = get_random_table_indexes(nbtable,10,len(nbtable))
	clset = {}
	for row in test:
		k = row[keycol]
		if k not in clset:
			clset[k] = run_single_test(k,nb_classify(row,attlist,rules),0,0)
		else:
			tmp = clset.get(k)
			clset[k] = run_single_test(row[keycol],nb_classify(row,attlist,rules),tmp[0],tmp[1])
	p = 0.0
	for k in clset.keys():
		tmp = clset.get(k)
		p += calculate_p(tmp[0],tmp[1])
	p = float(p) / float(10)
	se = calculate_stdE(p,len(test))
	return p, se

def run_single_test(act,pred,tp,t):
	t += 1
	if(act == pred):
		tp  += 1
	return tp,t

def calculate_p(tp,t):
	return float(tp) / float(t)

def calculate_stdE(p,t):
	return math.sqrt(float(p*(1-p)) / float(t))

def print_predAcc_format(fname,xs):
	p, stdE = average_p_and_stdE_list(xs)
	"""Prints name, prediction accuracy and std err"""
	print fname + ": p = " + str(p) + " +- " + str(stdE)
	#print "  " + fname +": p = " + '{:.3f}'.format(p) + " +- " + '{:.3f}'.format(stdE)

def average_p_and_stdE_list(xs):
	p = 0.0
	stdE = 0.0
	for x in xs:
		p += float(x[0])
		stdE += float(x[1])
	p = float(p) / float(len(xs))
	stdE = float(stdE) / float(len(xs))
	return p, stdE

def step5(table,atts):
	print '===========================================\nSTEP 5: Confusion Matricies \n==========================================='
	nb_v1 = init_nxn(10)
	nb_v2 = init_nxn(10)
	knn = init_nxn(10)
	lrn = init_nxn(10)
	kfold = k_folds(deepcopy(table),10)

	for f in kfold:
		training,test = holdout_partition(f)
		#s5_NB1_dict(nb_v1,deepcopy(training),deepcopy(test))
		#s5_NB2_dict(nb_v2,deepcopy(training),deepcopy(test))
		s5_LR_dict(lrn,deepcopy(training),deepcopy(test))
		s5_KNN_dict(knn, deepcopy(training), deepcopy(test))

	s5_print_confusion_table("Linear Reression",lrn)
	s5_print_confusion_table("K Nearest Neighbor", knn)
	#s5_print_confusion_table("Naive Bayes I",nb_v1)
	#s5_print_confusion_table("Naive Bayes II",nb_v2)

def s5_LR_dict(lrd,training,test):
	weights = []
	mpgs = []
	for row in training:
		weights.append(float(row[4]))
		mpgs.append(float(row[1]))
		_weight, _mpg, m = hw2.calculate_best_fit_line(weights, mpgs)

	act = []
	prd = []

	for row in test:
		act.append(hw2.get_mpg_rating(row[1]))
		prd.append(hw2.get_mpg_rating(get_linear_prediction(_weight, _mpg, m, x = float(row[4]))))
	update_cm(lrd,act,prd)

def s5_KNN_dict(thing, training, test):
	nnn = 5
	training = table_to_lick_dicts(training)
	test = table_to_lick_dicts(test)
	act = []
	prd = []
	for row in test:
		act.append(hw2.get_mpg_rating(row["MPG"]))
		prd.append(get_mpg_class_label(get_knn(row, nnn, training)))
	update_cm(thing, act, prd)

def s5_NB1_dict(mac, training,test):
	nbtable = temp_table_with_NHTSA_rating(training)
	ac, p = s5_run_NB(nbtable,test)
	update_cm(mac,ac,p)

def update_cm(mac,a,p):
	for i in range(len(a)):
		mac[a[i] -1 ][p[i] - 1] = mac[a[i] - 1][p[i] - 1] + 1

def s5_NB2_dict(mac, training,test):
	nbtable = build_table_with_gaussian(training,4)
	ac, p = s5_run_NB(nbtable,test)
	update_cm(mac,ac,p)

def s5_run_NB(nbtable,test):
	keycol = 1
	attlist = [2,3,4]
	rules = build_all_class_dicts(nbtable,keycol,attlist)
	rand_inst = get_random_table_indexes(nbtable,10,len(nbtable))
	pred = []
	actual = []
	for row in test:
		actual.append(hw2.get_mpg_rating(row[keycol]) )
		pred.append(int(float(nb_classify(row,attlist,rules))) )
	return actual,pred

def s5_knn_dict(knn,training,test):
	return None

def s5_print_confusion_table(name,table):
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

def init_nxn(n):
	l =[]
	for i in range(1,n+1):
		l.append([])
		for j in range(1,n+1):
			l[i - 1].append(0)
	return l

def step6(table,atts):
	print '===========================================\nSTEP 6: My Heart Will Go On \n==========================================='
	kfold = k_folds(deepcopy(table),10)
	nb = init_nxn(2)
	knn = init_nxn(2)

	for f in kfold:
		training,test = holdout_partition(f)
		s6_NB(nb,deepcopy(training),deepcopy(test))

	s6_print_confusion_table("Naive Bayes", nb)
	return None

def s6_NB(mac, training,test):
	ac, p = s6_run_NB(training,test)
	update_cm(mac,ac,p)

def s6_KNN(mac, training,test):
	return None

def s6_run_NB(nbtable,test):
	keycol = 3
	attlist = [0,1,2]
	rules = build_all_class_dicts(nbtable,keycol,attlist)
	rand_inst = get_random_table_indexes(nbtable,10,len(nbtable))
	pred = []
	actual = []
	for row in test:
		actual.append(row[keycol]) 
		pred.append(nb_classify(row,attlist,rules))
	return actual,pred


def s6_print_confusion_table(name,table):
	#table = make_table(file_name,summary_atts)
	print name + " - Stratified 10-Fold Cross Validation"
	headers = [" ","No","Yes","Total","Recognition (%)"]
	tmp_table = []
	tmp_table.append(["No"])
	tmp_table.append(["Yes"])

	print tabulate(tmp_table,headers,tablefmt="rst")

def build_conmat(a,p):
	result = init_nxn(n)
	for i in range(len(p)):
		if a[i] == p[i] and a [i] == "YES":
			None


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

def print_row(row):
	out = "instance: "
	for x in range(len(row) - 1):
		out += str(row[x]) + ", "
	out += str(row[len(row) - 1])
	print out

def get_random_table_indexes(table,n, size):
	index = sorted([int(random.random()*(size-1)) for _ in range(0, n)])
	result = []
	for i in index:
		result.append(table[i])
	return result


def main():
	step1("auto-data-cleaned.txt")
	step2("auto-data-cleaned.txt")
	atts,table = table_from_csv("auto-data-cleaned.txt")
	#rebuild_table_with_mpg_rating(table)
	step3(table,atts)
	step4(table,atts)
	step5(table,atts)
	tatts,ttable = table_from_csv("titanic.txt")
	#step6(ttable,tatts)
	return None

main()