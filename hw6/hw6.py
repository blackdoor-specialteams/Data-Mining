# coding=utf-8
import math
import lib
import copy
import itertools
from tabulate import tabulate


SUPPORT_DICT = dict()
MINSUP = .8

def hash_dict(d):
	h = ""
	keys = sorted(d.keys())
	for key in keys:
		h += str(key)
		h += str(d[key])
	return h

def get_support(itemset, dataset):
	global SUPPORT_DICT
	count = 0
	support = 0
	h = hash_dict(itemset)
	if(h in SUPPORT_DICT):
		support = SUPPORT_DICT[h]
	else:
		for entry in dataset:
			if subset(itemset, entry):
				count += 1
		support = count/float(len(dataset))
		SUPPORT_DICT[h] = support
	return support

def subset(subset, superset):
	for item_key in subset.keys():
		if subset[item_key] != superset[item_key]:
			return False
	return True

def get_confidence(itemset):
	return None

def get_completeness(itemset):
	return None

#deprecated
def itemsets_equal(a, b):
	if len(a) != len(b):
		return False
	for e in a:
		if e not in b:
			return False
	return True

def itemsets_union(a, b):
	u = copy.deepcopy(a)
	for key in b.keys():
		if key not in u:
			u[key] = b[key]
	return u

def get_L1(dataset, minsup = MINSUP):
	l1 = []
	for entry in dataset:
		for key in entry.keys():
			e = {key:entry[key]} # e is an itemset
			if e not in l1: 
				sup = get_support(e, dataset)
				#print e
				#print sup
				if(sup) >= minsup:
					l1.append(e)
	return l1

def num_common_items(a, b):
	count = 0
	for e in a.keys():
		if e in b and b[e] == a[e]:
			count += 1
	return count

def set_in_setofsets(_set, setofsets):
	for __set in setofsets:
		if __set == _set:
			return True
	return False

def get_Lk(last_l, dataset, minsup = MINSUP):
	candidates = get_candidate_set(last_l)
	Lk = []
	for candidate in candidates:
		subsets = get_subsets(candidate)
		supported = True
		for subset in subsets:
			if not set_in_setofsets(subset, last_l):
				supported = False
				break
		if supported:#if get_support(candidate, dataset) >= minsup:
			Lk.append(candidate)
	return Lk

def get_subsets(itemset):
	subsets = []
	for key in itemset:
		subset = copy.copy(itemset)
		subset.pop(key)
		subsets.append(subset)
	#print itemset
	#print subsets
	return subsets

def get_candidate_set(itemset):
	candidates = []
	for a in range(0,len(itemset)):
		for b in range(a,len(itemset)):
			if itemset[a] != itemset[b]:
				if(num_common_items(itemset[a], itemset[b]) == (len(itemset[a]) - 1)):
					union = itemsets_union(itemset[a],itemset[b])
					if(not set_in_setofsets(union, candidates)):
						candidates.append(union)
	#print candidates
	return candidates

def hw6():
	file_name = "agaricus-lepiota.txt"
	supp_sets = []
	dataset = lib.dataset_from_file(file_name)

	l = get_L1(dataset)
	while l:
		supp_sets += l
		l = get_Lk(l, dataset)
	for itemset in supp_sets:
		print itemset
	
'''
################################################################################
################################################################################
People's Republic of North Lead By Glorious Party Leader Nate
DMZ 214
Southern Provience of Code, True Heir to the Throne: CJ
################################################################################
################################################################################
'''
def bad_apriori(ds):
	"""
	"""
	min_conf = .8
	min_sup = .01
	values = get_all_att_values(ds)
	rules = []
	#get first generation of rules
	rules.append(supportedRules(init_Rules(values,len(ds)),ds,min_sup,min_conf))
	for x in rules[0]:
		print str(x)

def run_apriori(ds):
	"""
	"""
	min_conf = .8
	min_sup = .01
	FinalItemSets = {}
	c_set = get_base_Itemset(ds)
	length = len(ds)
	
	print c_set

def supportedItems(cset,ds,mS):
	return None

def init_Rules(values,length):
	"""
	Build the first set of rules. 
	"""
	result = []
	allkeys = values.keys()
	for k in allkeys:
		value_list = values[k]
		tmpkeys =  allkeys[:]
		tmpkeys.remove(k)
		for x in value_list:
			for t in tmpkeys:
				for element in itertools.product([x],values[t]):
					a = Rule(length)
					a.left = {k:element[0]}
					a.right = {t:element[1]}
					a.att_list.append(x)
					a.att_list.append(t)
					result.append(a)
	return result

def next_gen_Rules(values,rules):
	"""
	"""
	for k in values.keys():
		v = values[k]

def supportedRules(rules,ds,ms,mc):
	"""
	"""
	result = []
	for x in rules:
		for r in ds:
			x.checkAgainst(r)
	for x in rules:
		if x.support() >= ms and x.confidence() >= mc:
			result.append(x)
	return result

def get_all_att_values(ds):
	"""
	Returns a dictionary of lists of every value of an attribute in a dataset.
	"""
	keys = ds[0].keys()
	result = {}
	for k in keys:
		result[k] = []
		for x in ds:
			tmp = x[k]
			if tmp not in result[k]:
				result[k].append(tmp)
	return result

def get_base_Itemset(ds):
	"""
	Returns a dictionary of lists of every value of an attribute in a dataset.
		{ATTRIBUTE:{VALUE:COUNT}}
	"""
	result = {}
	keys = ds[0].keys()
	for x in keys:
		result[x] = {}
	for r in ds:
		for k in keys:
			tmp = r[k]
			if tmp not in result[k].keys():
				result[k][tmp] = 1
			else:
				result[k][tmp] += 1
	return result
=======

>>>>>>> 32f5ad827640e2df8e33b2666cdf7c23c9c35286

def s1_print_confusion_table():
	"""
	"""
	headers = []
	table = []
	print tabulate(tmp_table,headers,tablefmt="rst")

class Itemset:
	def __init__(self,t):
		self.items = {}
		self.att_list = []
		self.generation = t

class Rule:
	def __init__(self,t):
		self.left = {}
		self.right = {}
		self.att_list = []
		self.l_count = 0
		self.r_count = 0
		self.b_count = 0
		self.t_count = t
		self.generation = t

	def checkAgainst(self,inst):
		"""
		Checks an instance against the rule and increments the 
		relevant rule values. 
		"""
		isLeft = False
		isRight = False
		if self.isSubset(self.left,inst):
			self.l_count += 1
			isLeft = True
		if self.isSubset(self.right,inst):
			self.r_count += 1
			isRight = True
		if isLeft and isRight:
			self.b_count += 1

	def isSubset(self,d1,d2):
		"""
		Compares two dictionaries to see if d1 is a subset of d2.
		"""
		t1 = set(self.get_tupleList(d1))
		t2 = set(self.get_tupleList(d2))
		return t1.issubset(t2)

	def get_tupleList(self,d):
		"""
		Returns list of tuples of a dictionary's K,V pairs.
		"""
		tmp = []
		for k, v in d.items():
			tmp.append((k, v))
		return tmp

	def confidence(self):
		"""
		"""
		return float(self.b_count) / float(self.l_count)
	
	def support(self):
		"""
		"""
		return float(self.b_count) / float(self.t_count)
	
	def support(self):
		"""
		"""
		return float(self.b_count) / float(self.r_count)
	
	def lift(self):
		"""
		"""
		return float(self.b_count) / float(self.r_count)

	def __eq__(self, other):
		if isinstance(other, self.__class__):
			left = (self.left == other.left)
			right = (self.right == other.right)
			#reverse1 = ((self.left == other.right) and (self.right == other.left))
			return (left and right) 
			#or reverse1
		else:
			return False
	
	def __str__(self):
		out = "if "
		for x in self.left:
			out += str(x) + "=" + str(self.left[x]) + " "
		out += "=> "
		for x in self.right:
			out += str(x) + "=" + str(self.right[x]) + " "
		return out

#////////////////////////////////////////////////////////////////
def main():
	mushroom = "agaricus-lepiota.txt"
	titanic = "titanic.txt"

	ds = lib.dataset_from_file(titanic)
	run_apriori(ds)

if __name__ == '__main__':
	main()
