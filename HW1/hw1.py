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

def read_dataset(file_name):
	with open(file_name,"rb") as input_file:
		reader = csv.DictReader(input_file)

	return []

def count_instances(file_name):
	with open(file_name) as f:
		return sum(1 for _ in f)
	
def check_for_duplicates(dataset):
	return []
	
def print_dataset_info():
	return 0

def combine_two_datasets(data1,data2):
	return None

def step_two(filename):
	dataset = read_dataset(filename)
	
	print("--------------------------------------------------")
	print(filename)
	print("--------------------------------------------------")
	
	duplicates = check_for_duplicates(dataset)
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