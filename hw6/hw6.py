# coding=utf-8
import math
import lib
import copy
import itertools
from tabulate import tabulate


SUPPORT_DICT = dict()
SET_COUNTS = dict()
MINSUP = .8
MINCONF = .01

def hash_dict(d):
	h = ""
	keys = sorted(d.keys())
	for key in keys:
		h += str(key)
		h += str(d[key])
	return h

def intersect_is_nullset(itemset_a, itemset_b):
	for key in itemset_a.keys():
		if key in itemset_b:
			return False
	return True

def get_rule_candidates(supported_itemsets, all_itemsets, dataset):#, minsup = minsup, minconf = MINCONF):
	candidates = []
	for supported_itemset in supported_itemsets:
		for itemset in all_itemsets:
			if(intersect_is_nullset(supported_itemset, itemset)):
				#if get_rule_support(supported_itemset, itemset, dataset) >= minsup and get_confidence(supported_itemset, itemset, dataset) >= minconf:
				candidates.append((supported_itemset, itemset))
	return candidates

def filter_rule_candidates(candidates, dataset, minsup = MINSUP, minconf = MINCONF):
	out = []
	for candidate in candidates:
		if get_rule_support(candidate[0], candidate[1], dataset) >= minsup and get_confidence(candidate[0], candidate[1], dataset) >= minconf:
			out.append(candidate)
	return out

def get_item_sets(dataset, max_size = 10):
	sets = []
	for entry in dataset:
		for attrib in entry.keys():
			s = {attrib:entry[attrib]}
			if s not in sets:
				sets.append(s)
	l = get_candidate_set(sets)
	while l and len(l[0]) < max_size:
		sets += l
		l = get_candidate_set(l)
	return sets

def get_lift(rhs, lhs, dataset):
	return get_support(itemsets_union(lhs, rhs), dataset)/float(get_support(lhs, dataset)*get_support(rhs, dataset))

def get_rule_support(lhs, rhs, dataset):
	n = 0
	for entry in dataset:
		match = True
		for attrib in lhs.keys():
			if not (attrib in entry and lhs[attrib] == entry[attrib]):
				match = False
				break
		if match:
			for attrib in rhs.keys():
				if not(attrib in entry and rhs[attrib] == entry[attrib]):
					match = False
					break
		if match:
			n += 1
	return n/float(len(dataset))

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
		if item_key in superset:
			if subset[item_key] != superset[item_key]:
				return False
	return True

def get_confidence(lhs, rhs, dataset):
	global SET_COUNTS
	union_cached = False
	lhs_cached = False
	count_lhs = 0
	count_union = 0
	h_lhs = hash_dict(lhs)
	union = itemsets_union(lhs, rhs)
	h_union = hash_dict(union)
	if h_lhs in SET_COUNTS:
		lhs_cached = True
		count_lhs = SET_COUNTS[h_lhs]
	if h_union in SET_COUNTS:
		union_cached = True
		count_union = SET_COUNTS[h_union]
	if not lhs_cached or not union_cached:
		for entry in dataset:
			if not union_cached:
				if subset(union, entry):
					count_union += 1
			if not lhs_cached:
				if subset(lhs, entry):
					count_lhs += 1
		SET_COUNTS[h_lhs] = count_lhs
		SET_COUNTS[h_union] = count_union
	if count_union == 0 or count_lhs == 0:
		return 0
	return count_union/float(count_lhs)

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

def get_pretty_print_rule_row(rule, dataset):
	row = []
	lhs = ""
	rhs = ""
	for key in rule[0].keys():
		lhs += key
		lhs += "="
		lhs += rule[0][key]
		lhs += ", "
	lhs = lhs [0:-2]
	for key in rule[1].keys():
		rhs += key
		rhs += "="
		rhs += rule[1][key]
		rhs += ", "
	rhs = rhs [0:-2]
	row.append(lhs + " => " + rhs)
	row.append(str(get_rule_support(rule[0], rule[1], dataset))[0:5])
	row.append(str(get_confidence(rule[0], rule[1], dataset))[0:5])
	row.append(str(get_lift(rule[0], rule[1], dataset))[0:5])
	return row

def itemsets_intersect(a, b):
	intersect = dict()
	for key in a.keys():
		if key in b and b[key] == a[key]:
			intersect[key] = a[key]
	return intersect

def get_inverse_intersect(a, b):
	result = dict()
	union = itemsets_union(a, b)
	intersect = itemsets_intersect(a, b)
	for key in union.keys():
		if key not in intersect:
			result[key] = union[key]
	return result

def get_rules_from_itemset(itemset, dataset, rhs = {}, minconf = MINCONF):
	if len(itemset) == 1:
		return []
	rules = []
	lhss = get_subsets(itemset)
	for lhs in lhss:
		_rhs = itemsets_union(get_inverse_intersect(lhs, itemset), rhs)
		if len(_rhs) > 0 and get_confidence(lhs, _rhs, dataset) > minconf:
			rules.append((lhs,_rhs))
			rules += get_rules_from_itemset(lhs, dataset, _rhs)
	return rules

def hw6():
	file_name = "agaricus-lepiota.txt"
	supp_sets = []
	dataset = lib.dataset_from_file(file_name)

	l = get_L1(dataset)
	while l:
		supp_sets = l
		l = get_Lk(l, dataset)
	for itemset in supp_sets:
		print itemset
	#all_itemsets = get_item_sets(dataset)
	#rule_candidates = get_rule_candidates(supp_sets, all_itemsets, dataset)
	#rules = filter_rule_candidates(rule_candidates, dataset)
	rules = []
	for itemset in supp_sets:
		rules += get_rules_from_itemset(itemset, dataset)
	print ["rule", "support", "confidence", "lift"]
	for rule in rules:
		print get_pretty_print_rule_row(rule, dataset)
	
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
