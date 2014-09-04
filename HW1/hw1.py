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

def readDataset(filename):
	file = open(filename,"rb")
	reader = csv.reader(file)

	return []

def countInstances():
	return 0
	
def checkForDuplicates():
	return []
	
def printDatasetInfo():

def combineTwoDatasets(data1,data2):


def stepTwo(filename):
	dataset = readDataset(filename)
	
	print("--------------------------------------------------")
	print(filename)
	print("--------------------------------------------------")
	
	duplicates = checkForDuplicates(dataset)
	instcount = countInstances(dataset)
	
	print("No. of instances: ", instcount)
	print("Duplicates: ", duplicates )
	
def stepThree():

def stepFour():

def main():
	stepTwo("auto-mpg.txt")
	
	raw_input("Hit Enter to EXIT")
main()