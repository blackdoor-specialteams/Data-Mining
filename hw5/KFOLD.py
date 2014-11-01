import copy
import random 

def k_folds(dataset,k):
	rdm = copy.deepcopy(dataset) 
	n = len(dataset)
	step = n / k
	for i in range(n):
		j = random.randint(0,n-1)
		rdm[i], rdm[j] = rdm[j],rdm[i]
	return [rdm[i:i + step] for i in range(0, len(rdm), step)]
	