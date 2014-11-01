import math

"""
###################################################################################################
			K-NN Classifier
###################################################################################################
How to Use:

"""
def knn_classify():
	return None

def get_knn(instance, k, training_set, attribs):
	'''
	Returns the k nearest neighbors to instance based on training_file and attribs
	'''
	ranges = get_ranges(training_set, attribs)
	distances = []
	for inst in training_set:
		distances.append( [get_neighbor_d(instance, inst, ranges, attribs), inst] )
	distances = sorted(distances, key=itemgetter(0))
	ret = [e[1] for e in distances[0:k]]
	return ret

def get_neighbor_d(instance, neighbor, ranges, attribs):
	"""
	Returns the distance between instance and neighbor.
	The distance between two instances is the sum of the Euclidean distances between the normalized values of each attribute in attribs. Non-Ordered attributes have a distance 1 if their values are equal, else 0.
	"""
	d = 0
	for attrib in attribs:
		if float(ranges[attrib]) == 0:
			None
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
		try:
			for inst in data_set:
					xs.append(float(inst[attrib]))
			ranges[attrib] = max(xs) - min(xs)
		except ValueError:
			ranges[attrib] = 1.0
	return ranges

def get_euclidean_d(i, i2):
	"""
	Returns the Euclidean distance between i and i2
	"""
	return float(math.sqrt(pow((i-i2),2)))