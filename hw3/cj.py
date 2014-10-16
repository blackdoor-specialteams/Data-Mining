# coding=utf-8
import math
import hw2
import hw3
import csv
import random
from copy import deepcopy
from operator import itemgetter
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
	rand_inst = get_random_indexes(nbtable,5,len(nbtable))
	for row in rand_inst:
		print_instance(row)
		print_classification(row,rules,keycol,attlist)

def nb_v2(table,attlist,keycol):
	nbtable = build_table_with_gaussian(table,4)
	rules = build_all_class_dicts(nbtable,keycol,attlist)
	rand_inst = get_random_indexes(nbtable,5,len(nbtable))
	for row in rand_inst:
		print_instance(row)
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
		nb_v1.append(s4_NB_v1(deepcopy(training),deepcopy(test)))
		nb_v2.append(s4_NB_v2(deepcopy(training),deepcopy(test)))
		lnr.append(s4_LR(deepcopy(training),deepcopy(test)))
		knn.append()

	print_predAcc_format("Linear Reression",lnr)
	print_predAcc_format("Naive Bayes I",nb_v1)
	print_predAcc_format("Naive Bayes II",nb_v2)

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

	print_predAcc_format("Linear Reression",lnr)
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
	training = hw3.table_to_lick_dicts(training)
	test = hw3.table_to_lick_dicts(test)
	predictions = []
	acutals = []
	for instance in test:
		None

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
		prd = str(hw2.get_mpg_rating(hw3.get_linear_prediction(_weight, _mpg, m, x = float(row[4]))))
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
	rand_inst = get_random_indexes(nbtable,10,len(nbtable))
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
	print fname
	print p
	print stdE
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
		s5_NB1_dict(nb_v1,deepcopy(training),deepcopy(test))
		s5_NB2_dict(nb_v2,deepcopy(training),deepcopy(test))
		s5_LR_dict(lrn,deepcopy(training),deepcopy(test))

	s5_print_confusion_table("Linear Reression",lrn)
	s5_print_confusion_table("Naive Bayes I",nb_v1)
	s5_print_confusion_table("Naive Bayes II",nb_v2)

def s5_LR_dict(lrd,training,test):
	weights = []
	mpgs = []
	for row in training:
		weights.append(int(row[4]))
		mpgs.append(int(row[1]))
		_weight, _mpg, m = hw2.calculate_best_fit_line(weights, mpgs)
	#print _weight
	#print _mpg
	#print m
	act = []
	prd = []
	for row in test:
		act.append(int(float(row[1])))
		prd.append(hw2.get_mpg_rating(hw3.get_linear_prediction(_weight, _mpg, m, x = int(row[4]))))
	update_cm(lrd,act,prd)


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
	rand_inst = get_random_indexes(nbtable,10,len(nbtable))
	pred = []
	actual = []
	for row in test:
		actual.append(int(float(row[keycol]) ))
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
	headers = headers + ["Total", "Recognition"]
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
#		row.append("{:.5f}".format(rec))
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
	rand_inst = get_random_indexes(nbtable,10,len(nbtable))
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

def build_gen_conmat(n,a,p):
	result = init_nxn(n)
	outcomes = {}


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

def main():
	atts,table = table_from_csv("auto-data-cleaned.txt")
	#rebuild_table_with_mpg_rating(table)
	#step3(table,atts)
	step4(table,atts)
	step5(table,atts)
	tatts,ttable = table_from_csv("titanic.txt")
	#step6(ttable,tatts)

main()