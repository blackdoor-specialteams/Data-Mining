# coding=utf-8
import math
import lib
import itertools
from tabulate import tabulate

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
