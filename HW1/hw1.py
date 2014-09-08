"""
@Names: Cj Buresch and Nathan Fischer
@Date: 9/3/2014
@ Homework #1 -- Dataset Manipulation
@Description: 
	Reads datasets from file, prints the number of instances from each. 
	Combine the two datasets into one and print the result. Resolves any
	missing attribute data for instances accordingly. LAstly, computes 
	summary data from the combined datasets. 
@Version: Python v2.7
"""
import csv

def get_dataset(file_name):
	""" Reads a csv file and returns a table as a list of lists"""
	the_file = open(file_name)
	the_reader = csv.reader(the_file, dialect = 'excel')
	table = []
	for row in reader:
		if len(row) > 0:
			table.append(row)
	return table
	
#returns a list of duplicates	
#keys are attributes to be considered part of the key
def check_for_duplicates(file_name, keys = ()):
	dupes = []
	instances = []
	with open(file_name,"r") as input_file:
		reader = csv.DictReader(input_file)
		for instance in reader:
			if not keys:
				keys = instance.keys()
			key = get_key_from_attribs(instance, keys)
			if key in instances and key not in dupes:
				dupes.append(key)
			else:
				instances.append(key)
	return dupes
	
def count_instances(file_name):
	with open(file_name, "r") as f:
		return sum(1 for _ in f)
	
def print_csv(table):
	"""Prints a table in csv format"""
	for row in table:
		for i in range(len(row)-1):
			sys.stdout.write(str(row[i])+ ',')
		print row[-1]

def print_dataset_info():
	return 0

def get_key_from_attribs(instance, attribs = ('model_year', 'car_name')):
	key = ''
	for attrib in attribs:
		key += str(instance[attrib]) + " "
	key = key[0:-1] # trim trailing space
	return key

def join_into_dict_list(in_files = ('auto-prices.txt', 'auto-mpg.txt') , out = "", keys = ('model_year', 'car_name')):
#	joined = dict()
#	for input_file in in_files:
#		with open(input_file, 'r') as f:
#			reader = csv.DictReader(input_file)
#			for instance in reader:
#				print(instance)
#				key = get_key_from_attribs(instance, keys)
#				if key in joined:
#					joined[key].update(instance)
#				else:
#					joined[key] = instance
	joined = []
	for left_file in in_files:
		with open(left_file, 'r') as f:
			l_reader = csv.DictReader(f)
			for l_instance in l_reader:
				joined_inst = dict()
				for right_file in in_files:
					if right_file != left_file:
						with open(right_file, 'r') as rf:
							r_reader = csv.DictReader(rf)
							for r_instance in r_reader:
								match = True
								for key in keys:
									if not (r_instance[key] == l_instance[key]):
										match = False
										break
								if match:
									joined_inst.update(r_instance)
									joined_inst.update(l_instance)
				




	return joined

def combine_two_datasets(data1,data2):
	return None

def step_two(filename):
	#dataset = read_dataset(filename)
	
	print "--------------------------------------------------"
	print str(filename)
	print "--------------------------------------------------"
	
	duplicates = check_for_duplicates(filename, ["model_year", "car_name"])
	inst_count = count_instances(filename)
	
	print "No. of instances: " + str(inst_count)
	print "Duplicates: " + str(duplicates)
	
def step_three():
	return None

def step_four():
	return None

def main():
	step_two("auto-mpg.txt")
	step_two("auto-prices.txt")
	raw_input("Hit Enter to EXIT")

main()
