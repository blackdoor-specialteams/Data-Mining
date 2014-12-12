# coding=utf-8

import math
import lib
import copy

SUPPORT_DICT = dict()

def get_support(itemset, dataset):
	count = 0
	support = 0
	if(itemset in SUPPORT_DICT):
		support = SUPPORT_DICT[itemset]
	else:
		for entry in dataset:
			all_in = True
			entry_list = entry.values()
			for item in itemset:
				if item not in entry_list:
					all_in = False
					break
			if all_in:
				count += 1
		support = count/float(len(dataset))
		SUPPORT_DICT[itemset] = support
	return support

def get_confidence(itemset):
	return None

def get_completeness(itemset):
	return None

def itemsets_equal(a, b):
	if len(a) != len(b):
		return False
	for e in a:
		if e not in b:
			return False
	return True

def itemsets_union(a, b):
	u = copy.deepcopy(a)
	for item in b:
		if item not in u:
			u.append(item)
	return u

def get_L1(dataset, minsup = .5):
	l1 = []
	for entry in dataset:
		for e in entry.values():
			if e not in l1 and get_support([e], dataset) >= minsup:
				l1.append([e])
	return l1

def num_common_items(a, b):
	count = 0
	for e in a:
		if e in b:
			count += 1
	return count

def set_in_setofsets(_set, setofsets):
	for __set in setofsets:
		if itemsets_equal(__set, _set):
			return True
	return False

def get_Lk(last_l, dataset, minsup = .5):
	candidates = get_candidate_set(last_l)
	Lk = []
	for candidate in candidates:
		if get_support(candidate, dataset) >= minsup:
			Lk.append(candidate)
	return Lk

def get_candidate_set(itemset):
	candidates = []
	for a in last_l:
		for b in last_l:
			if not itemsets_equal(a,b):
				if(num_common_items(a, b) = len(a) - 1):
					union = itemsets_union(a,b)
					if(set_in_setofsets(union, candidates)):
						candidates.append(union)
	return candidates

def hw6():
	file_name = "titanic.txt"
	rules = []
	dataset = lib.dataset_from_file(file_name)

	l = get_L1(dataset)
	while l:
		rules += l
		l = get_Lk(l, dataset)
	

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
