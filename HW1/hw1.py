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

#fundamentally broken
def get_dataset_reader(file_name):
	with open(file_name,"r") as input_file:
		reader = csv.DictReader(input_file)
	return reader

def count_instances(file_name):
	with open(file_name, "r") as f:
		return sum(1 for _ in f)
	
#returns a list of duplicates	
#keys are attributes to be considered part of the key
def check_for_duplicates(file_name, keys = ('model_year', 'car_name')):
	dupes = []
	instances = []
	with open(file_name,"r") as input_file:
		reader = csv.DictReader(input_file)
		for instance in reader:
			key = get_key_from_attribs(instance, keys)
			if key in instances and key not in dupes:
				dupes.append(key)
			else:
				instances.append(key)
	return dupes
	
def print_dataset_info():
	return 0

def get_key_from_attribs(instance, attribs = ('model_year', 'car_name')):
	key = ''
	for attrib in attribs:
		key += str(instance[attrib]) + " "
	key = key[0:-1] # trim trailing space
	return key

def join_into_file(in_files = ('auto-prices.txt', 'auto-mpg.txt') , out = "", keys = ('model_year', 'car_name')):
	joined = dict()
	for input_file in in_files:
		with open(input_file, 'r') as f:
			reader = csv.DictReader(input_file)
			for instance in reader:
				print(instance)
				key = get_key_from_attribs(instance, keys)
				if key in joined:
					joined[key].update(instance)
				else:
					joined[key] = instance
	print(joined)

def combine_two_datasets(data1,data2):
	return None

def step_two(filename):
	#dataset = read_dataset(filename)
	
	print("--------------------------------------------------")
	print(filename)
	print("--------------------------------------------------")
	
	duplicates = check_for_duplicates(filename)
	inst_count = count_instances(filename)
	
	print("No. of instances: ", inst_count)
	print("Duplicates: ", duplicates )
	
def step_three():
	return None

def step_four():
	return None

def main():
	step_two("auto-mpg.txt")

	input("Hit Enter to EXIT")

main()