# coding=utf-8
import csv
import re
#import random
import math
import TDIDT
import lib
import operator
from tabulate import tabulate

def auto_dataset_from_file(filename):
	"""Returns a  list of dictionaries from the file"""
	result = []
	with open(filename, 'r') as f:
		reader = csv.DictReader(f)
		for inst in reader:
			inst["Weight"] = lib.get_NHTSA_rating(inst["Weight"])
			inst['MPG'] = lib.get_mpg_rating(inst['MPG'])
			result.append(inst)
	return result

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
	tree = build_tree_from_dataset(ddict,"survived",[])
	kfold = lib.k_folds(ddict,10)
	tdidt_cm = lib.init_nxn(2)

	for i in range(10):
		training =[]
		for f in range(10):
			if f != i:
				training = training + kfold[f]
		test = kfold[i]
		s1_tree_run(tdidt_cm,training,test)

	s1_print_confusion_table("TDIDT", tdidt_cm)

def s1_tree_run(cm,traning,test):
	"""
    Runs tree over a training and a test set
    """
	key = "survived"
	tree = build_tree_from_dataset(traning,key,[])
	tree = tree.condense(tree)
	pred = []
	actual = []
	for row in test:
		actual.append(row[key]) 
		pred.append(tree.classify(row))
	s1_update_cm(cm,actual,pred)

def s1_update_cm(cm,a,p):
	for i in range(len(a)):
		cm[YN_to_int(a[i])][YN_to_int(p[i])] += 1

def s1_print_confusion_table(name,cm):
	"""prints a confusion matrix for titanic using tabulate"""
	#table = make_table(file_name,summary_atts)
	print name + " - Stratified 10-Fold Cross Validation"
	headers = ["Survived","No","Yes","Total","Recognition (%)"]
	tmp_table = []
	for i in range(0,2): 
		row = [int_to_YN(i)]
		total = float(sum(cm[i]))
		if total > 0:
			rec = float(cm[i][i]) / total
		else:
			rec = 0
		for x in cm[i]:
			row.append(str(x))
		row.append(str(sum(cm[i])))
		row.append(str(rec * 100))
		tmp_table.append(row)
	print tabulate(tmp_table,headers,tablefmt="rst")

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
	print '===========================================\nSTEP 2: TDIDT -- Auto Data \n========================================='
	ds = auto_dataset_from_file(filename)
	kfold = lib.k_folds(ds,10)
	tdidt_cm = lib.init_nxn(10)

	for i in range(10):
		training =[]
		for f in range(10):
			if f != i:
				training = training + kfold[f]
		test = kfold[i]
		s2_tree_run(tdidt_cm,training,test)

	lib.print_confusion_table("TDIDT", tdidt_cm)

def s2_tree_run(cm,training,test):
	key = "MPG"
	wrap = ["Model Year","Cylinders","Weight"]
	tree = build_tree_from_dataset(training,key,wrap)
	#tree = tree.condense(tree)
	pred = []
	actual = []
	for row in test:
		actual.append(row[key])
		pred.append(tree.classify(row))
	lib.update_cm(cm,actual,pred)

def step3(filename):
	print '===========================================\nSTEP 3: TDIDT -- Split Point \n========================================='
	ds = dataset_from_file(filename)
	adjust_ds_with_mpg(ds)
	kfold = lib.k_folds(ds,10)
	tdidt_cm = lib.init_nxn(10)

	for i in range(10):
		training =[]
		for f in range(10):
			if f != i:
				training = training + kfold[f]
		test = kfold[i]
		s3_tree_run(tdidt_cm,training,test)
	lib.print_confusion_table("TDIDT", tdidt_cm)

def s3_tree_run(cm,training,test):
	key = "MPG"
	split = calulate_Split(training)
	adjust_dataset_with_split(training,split)
	wrap = ["Model Year","Cylinders","Weight"]
	tree = build_tree_from_dataset(training,key,wrap)
	#tree = tree.condense(tree)
	pred = []
	actual = []
	adjust_dataset_with_split(test,split)
	for row in test:
		actual.append(row[key])
		pred.append(tree.classify(row))
	lib.update_cm(cm,actual,pred)


def adjust_dataset_with_split(ds,split):
	for x in ds:
		if x["Weight"] > split:
			x["Weight"] = 1
		else:
			x["Weight"] = 0

def adjust_ds_with_mpg(ds):
	for x in ds:
		x["MPG"] = lib.get_mpg_rating(x["MPG"])

def calulate_Split(ds):
	vs = []
	En = {}
	for x in ds:
		#print x
		vs.append(x["Weight"])
	sorted(vs)
	for v in vs:
		En[calculate_En_spliiter(ds,v)] = v
	return min(En.iteritems(), key=operator.itemgetter(0))[1]

def calculate_En_spliiter(ds,v):
	high = []
	low = []
	for x in ds:
		if x["Weight"] > v:
			high.append(x)
		else:
			low.append(x)
	return  (.1 * calculate_E_splitter(high)) + (.1 * calculate_E_splitter(low))

def calculate_E_splitter(ds):
	xs = {}
	E = 0.0
	count = 0
	for inst in ds:
			if inst["MPG"] not in xs.keys():
				xs[inst["MPG"] ] = 1
			else:
				xs[inst["MPG"]] += 1
			count+= 1
	for x in xs:
		p = float(xs[x])/float(count)
		E -= (p * math.log(p,2))
	return E

def step4(filename1,filename2):
	print '===========================================\nSTEP 4: TDIDT -- Printed Rules\n========================================='
	ds = dataset_from_file(filename1)
	print "--------------------------------------------\nSTEP 1"
	t = build_tree_from_dataset(ds,"survived",[])
	#t = t.condense(t)
	t.print_rules()

	print "--------------------------------------------\nSTEP 2"
	ds = auto_dataset_from_file(filename2)
	wrap = ["Model Year","Cylinders","Weight"]
	t = build_tree_from_dataset(ds,"MPG",wrap)
	#t = t.condense(t)
	t.print_rules()
	print "--------------------------------------------\nSTEP 3"
	ds = dataset_from_file(filename2)
	adjust_ds_with_mpg(ds)
	split = calulate_Split(ds)
	adjust_dataset_with_split(ds,split)
	t = build_tree_from_dataset(ds,"MPG",wrap)
	#t = t.condense(t)
	t.print_rules()

#//////////////////////////////////////////////////

def build_tree_from_dataset(ds,target,wrap):
	t = TDIDT.TDIDT(target)
	t.put_dataset(ds,wrap)
	#print t
	return t

def get_Att_En(ds,target):
	keys = ds[0].keys()
	result = {}
	for a in keys:
		result[a] = calculate_En(ds,a)
	return result

def calculate_En(ds,a,target):
	val = {}
	result = 0
	for x in ds:
		if x[a] not in val:
			val[a] = 1
		else:
			val[a] += 1
	total = sum(val.values())
	for v in val.keys():
		result += (float(val[v])/float(total)) * calculate_E(ds,a,v,target)
	return result

def calculate_E(ds,a,v,target):
	cls = {}
	result = 0
	for x in ds:
		if (x[target] not in cls) and (x[a] == v):
			cls[x[target]] = 1
		elif x[a] == v:
			cls[x[target]] += 1
	total = sum(cls.values())
	for v in cls.keys():
		result -= (float(cls[v])/float(total)) * math.log((float(cls[v])/float(total)),2)
	return result

def dataset_from_file(filename):
	"""Returns a  list of dictionaries from the file"""
	result = []
	with open(filename, 'r') as f:
		reader = csv.DictReader(f)
		for row in reader:
			result.append(row)
	return result

def main():
	dataset1 = "titanic.txt"
	dataset2 = "auto-data-cleaned.txt"
	step1(dataset1)
	step2(dataset2)
	step3(dataset2)
	step4(dataset1,dataset2)
	
if __name__ == '__main__':
    main()
