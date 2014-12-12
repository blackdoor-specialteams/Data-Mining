# coding=utf-8

import math
import lib
import copy

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


#////////////////////////////////////////////////////////////////
def main():
	mushroom = "agaricus-lepiota.txt"
	tictactoe = "tic-tac-toe.txt"

if __name__ == '__main__':
    main()
