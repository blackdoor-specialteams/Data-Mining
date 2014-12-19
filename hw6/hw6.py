# coding=utf-8
# authors@ Nathan Fischer, Christopher Buresch
# file@ hw6.py
# date@ 12/15/2014
import math
import lib
import copy
import itertools
from tabulate import tabulate

SUPPORT_DICT = dict()
SET_COUNTS = dict()
MINSUP = .7
MINCONF = .9

def hash_dict(d):
	"""
	Get a unique identifier for a dictionary 
	"""
	h = ""
	keys = sorted(d.keys())
	for key in keys:
		h += str(key)
		h += str(d[key])
	return h

def intersect_is_nullset(itemset_a, itemset_b):
	"""
	The intersection of a set is null 
	"""
	for key in itemset_a.keys():
		if key in itemset_b:
			return False
	return True

def get_rule_candidates(supported_itemsets, all_itemsets, dataset):
	"""
	Get all rule canadiates from list of supported itemsets 
	"""
	candidates = []
	for supported_itemset in supported_itemsets:
		for itemset in all_itemsets:
			if(intersect_is_nullset(supported_itemset, itemset)):
				candidates.append((supported_itemset, itemset))
	return candidates

def filter_rule_candidates(candidates, dataset, minsup = MINSUP, minconf = MINCONF):
	"""
	Return a list of supported itemsets
	"""
	out = []
	for candidate in candidates:
		if get_rule_support(candidate[0], candidate[1], dataset) >= minsup and get_confidence(candidate[0], candidate[1], dataset) >= minconf:
			out.append(candidate)
	return out

def get_item_sets(dataset, max_size = 10):
	"""
	Get all itemsets. 
	"""
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
	"""
	Return the calculated Lift for a rule
	"""
	return get_support(itemsets_union(lhs, rhs), dataset)/float(get_support(lhs, dataset)*get_support(rhs, dataset))

def get_rule_support(lhs, rhs, dataset):
	"""
	Return the calculated suport for the rule
	"""
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
	"""
	Return the support of a itemset 
	"""
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
	"""
	Returns if it is a subset of a another set
	"""
	for item_key in subset.keys():
		if item_key in superset:
			if subset[item_key] != superset[item_key]:
				return False
	return True

def get_confidence(lhs, rhs, dataset):
	"""
	Return rule confidence
	"""
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

def itemsets_union(a, b):
	"""
	Returns the set union
	"""
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
				if(sup) >= minsup:
					l1.append(e)
	return l1

def num_common_items(a, b):
	"""
	Count of items in common in sets
	"""
	count = 0
	for e in a.keys():
		if e in b and b[e] == a[e]:
			count += 1
	return count

def set_in_setofsets(_set, setofsets):
	"""
	Checks to see if set is in a set of sets 
	"""
	for __set in setofsets:
		if __set == _set:
			return True
	return False

def get_Lk(last_l, dataset, minsup = MINSUP):
	"""
	Return supported Lk generation rules 
	"""
	candidates = get_candidate_set(last_l)
	Lk = []
	for candidate in candidates:
		subsets = get_subsets(candidate)
		supported = True
		for subset in subsets:
			if not set_in_setofsets(subset, last_l):
				supported = False
				break
		if supported:
			Lk.append(candidate)
	return Lk

def get_subsets(itemset):
	"""
	Return subsets of a set
	""" 
	subsets = []
	for key in itemset:
		subset = copy.copy(itemset)
		subset.pop(key)
		subsets.append(subset)
	return subsets

def get_candidate_set(itemset):
	"""
	Get the suported Candidate sets 
	"""
	candidates = []
	for a in range(0,len(itemset)):
		for b in range(a,len(itemset)):
			if itemset[a] != itemset[b]:
				if(num_common_items(itemset[a], itemset[b]) == (len(itemset[a]) - 1)):
					union = itemsets_union(itemset[a],itemset[b])
					if(not set_in_setofsets(union, candidates)):
						candidates.append(union)
	return candidates

def tabulate_rules(rules,dataset):
	"""
	Print the rules "pretty" style 
	"""
	headers =  ["association rule","support","confidence","lift"]
	inst = []
	for rule in rules:
		tmp = get_pretty_print_rule_row(rule,dataset)
		if float(tmp[1]) >= MINSUP:
			inst.append(tmp)
	print tabulate(inst,headers,tablefmt="rst")


def get_pretty_print_rule_row(rule, dataset):
	"""
	Helper, formats for "pretty" printing
	"""
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
	"""
	Return the set intersection 
	"""
	intersect = dict()
	for key in a.keys():
		if key in b and b[key] == a[key]:
			intersect[key] = a[key]
	return intersect

def get_inverse_intersect(a, b):
	"""
	Return the "inverse intersect"
	"""
	result = dict()
	union = itemsets_union(a, b)
	intersect = itemsets_intersect(a, b)
	for key in union.keys():
		if key not in intersect:
			result[key] = union[key]
	return result

def get_rules_from_itemset(itemset, dataset, rhs = {}, minconf = MINCONF):
	"""
	Get the rules for an itemset 
	"""
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

def hw6(file_name):
	"""
	Find all supported rules for the dataset
	"""
	supp_sets = []
	dataset = lib.dataset_from_file(file_name)
	l = get_L1(dataset)
	while l:
		supp_sets = l
		l = get_Lk(l, dataset)
	rules = []
	for itemset in supp_sets:
		rules += get_rules_from_itemset(itemset, dataset)
	tabulate_rules(rules, dataset)

#////////////////////////////////////////////////////////////////

def main():
	mushroom = "agaricus-lepiota.txt"
	titanic = "titanic.txt"

	hw6(titanic)
	hw6(mushroom) # takes forever to run.... so careful. 

if __name__ == '__main__':
	main()
