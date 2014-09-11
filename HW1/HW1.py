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
import re
import os
from tabulate import tabulate

def get_dataset(file_name):
	""" Reads a csv file and returns a table as a list of lists"""
	the_file = open(file_name)
	the_reader = csv.reader(the_file, dialect = 'excel')
	table = []
	for row in reader:
		if len(row) > 0:
			table.append(row)
	return table
	
def check_for_duplicates(file_name, keys = ()):
	"""" Generates a list of duplicates from a csv using specified keys"""
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

def clean_data(filename_in,filename_out):
	""" Takes a CSV file, removes any row that has no MPG value"""
	with open(filename_in, 'rb') as _in, open(filename_out, "wb") as out:
		f1 = csv.reader(_in)
		writer = csv.writer(out)
		valid = []
		for row in f1:
			if row[2] != 'NA':
				writer.writerow(row)
	
def count_instances(file_name):
	"""Counts the number of rows or instances in a csv file"""
	with open(file_name, "r") as f:
		return sum(1 for _ in f)

def delete_dups_from_csv(filename_in,filename_out):
	""" Takes a CSV file, writes a second file that omits all duplicates"""
	with open(filename_in, 'rb') as _in, open(filename_out, "wb") as out:
		f1 = csv.reader(_in)
		writer = csv.writer(out)
		unique_rows = []
		for row in f1:
			if row not in unique_rows:
				writer.writerow(row)
				unique_rows.append(row)

def populate_empty_values(filein):
	""" Finds null values in a csv file and replaces them with the average for that attribute"""
	attribute_list = [0,1,3,4,5,7,9]

	for x in attribute_list:
		datalist = get_att_list(filein,x)
		avg = round(average(datalist),2)
		replace_all_NA_for_att(filein,x,avg)

def replace_all_NA_for_att(filein,index,avg):
	"""Reads the file in, iterates through every row and replaces NA in relevant row index with 
	the average."""
	writerows = []
	
	with open(filein, 'rb') as _in:
		f1 = csv.reader(_in)
		writerows.append(f1.next())
		for row in f1:
			if row[index] == 'NA':
				row[index] = avg
			writerows.append(row)

	with open(filein, "wb") as out:
		writer = csv.writer(out)
		for x in writerows:
			writer.writerow(x)

def print_summary_stats(file_name):
	""" Prints a 5 number summary of each attribute using tabulate"""
	print "Summary Stats:"

	headers = ["attribute","min","max","mid","avg","med"]
	summary_atts = [0,1,3,4,5,7,9]
	table = make_table(file_name,summary_atts)

	print tabulate(table,headers,tablefmt="rst")

def make_table(filein,summarylist):
	""" Builds a table from a csv file, with a set of attributes to include"""
	table = []
	for x in summarylist:
		att_list = get_att_list(filein,x)
		statlist = summerize_data(att_list)
		t = (get_attribute_name(filein,x),statlist[0],statlist[1],statlist[2],statlist[3],statlist[4])
		table.append(t)
		#print_att_summary_row(get_attribute_name(file_name,x),statlist)
	return table

def get_att_list(filein,index):
	""" Collects all attribute data from a csv file for a single attribute"""
	att_list = []
	with open(filein, 'rb') as _in:
		f1 = csv.reader(_in)
		next(f1)
		for row in f1:
			if row[index] != 'NA':
				att_list.append(float(row[index]))
	return att_list

def get_attribute_name(filein,index):
	""" gets label for an attribute at a certain index"""
	with open(filein, 'rb') as _in:
		f1 = csv.reader(_in)
		line1 = f1.next()
	return line1[index]

def summerize_data(attlist):
	""" Returns a list of the 5 number summary based on the provided list"""
	statlist = []

	lmin = min(attlist)
	lmax = max(attlist)
	lmid = midpoint(lmin,lmax)
	lavg = average(attlist)
	lmed = median(attlist)

	statlist.append(round(lmin,2))
	statlist.append(round(lmax,2))
	statlist.append(round(lmid,2))
	statlist.append(round(lavg,2))
	statlist.append(round(lmed,2))

	return statlist

def median(alist):
	""" Returns the median of a list"""
        srtd = sorted(alist) 
        mid = len(alist)/2   
        if (len(alist) % 2) == 0:  
                return (srtd[mid-1] + srtd[mid]) / 2.0
        else:
                return srtd[mid]

def average(numbers): 
	""" Finds the average for a list"""
	return float(sum(numbers))/len(numbers)

def midpoint(a,b):
	""" Finds the midpoint for a list"""
	return (a + b) / 2.0

def get_key_from_attribs(instance, attribs = ('model_year', 'car_name')):
	key = ''
	for attrib in attribs:
		key += str(instance[attrib]) + " "
	key = key[0:-1] # trim trailing space
	return key

def add_na(d, keys):
	na = 'NA'
	for key in keys:
		if key not in d:
			d[key] = na
	return d

def get_csv_line(d, keyset):
	line = ''
	for key in keyset:
		line += str(d[key]) + ','
	return line[0:-1] + '\n'

def write_csv_line(fileinst, d, keyset):
	fileinst.write(get_csv_line(d, keyset))

def write_csv_header(fileinst, keyset):
	line = ''
	for key in keyset:
		line += str(key) + ','
	fileinst.write(line[0:-1] + '\n')

def resolve_price_but_no_mpg_cases(file_name, output):
	keyset = []
	with open(output, 'wb') as wf:
		with open(file_name,'r') as f:
			reader = csv.DictReader(f)
			for inst in reader:
				if not keyset:
					keyset = inst.keys()
					write_csv_header(wf, keyset)
				if inst['msrp'] != 'NA' and inst['mpg'] == 'NA':
					print inst['car_name'] + ' ' + inst['model_year'] + ' ' + inst['msrp'] + ' ' + inst['mpg']
				else:
					write_csv_line(wf, inst, keyset)

def join_into_file( l_file, r_file, out, keys = ('model_year', 'car_name')):
	match = True
	l_keyset = []
	r_keyset = []
	keyset = []
	rejects = dict()
	with open(out, 'wb') as of, open(l_file, 'r') as lf:
		l_reader = csv.DictReader(lf)
#for each instance in the left table
		for l_inst in l_reader:
			if not l_keyset: l_keyset = l_inst.keys()
			any_match = False
			#print l_inst
			with open(r_file, 'r') as rf:
				r_reader = csv.DictReader(rf)
#for each instance in the right table
				for r_inst in r_reader:
					if not r_keyset: 
						r_keyset = r_inst.keys()
						keyset += l_keyset
						for key in r_keyset:
							if key not in keyset:
								keyset.append(key)
#write column labels for file
						write_csv_header(of, keyset)
					#print r_inst
					match = True
#for each key to join on, check if it is a match
					for key in keys:
						match &= l_inst[key] == r_inst[key]
#if instances should be joined, write them to the file
					if match:
						any_match = True
						r_inst.update(l_inst)
						#print 'match\n'
						#print 'join' + str(r_inst) + '\n'
						write_csv_line(of, r_inst, keyset)#joined.append(r_inst)
#else add the right instance to a set of non-matches
					else:
						#write_csv_line(of, add_na(l_inst, r_inst.keys()), keyset)
						#write_csv_line(of, add_na(r_inst, l_inst.keys()), keyset)
						rejects[r_inst['model_year']+r_inst['car_name']] = r_inst
#if there were no matches for the left instance, write it to the file with NA params
			if not any_match:
				write_csv_line(of, add_na(l_inst, r_inst.keys()), keyset)
		with open(out, 'rb') as matches:
			matches = matches.read()
#for each non-match
		for reject in rejects.itervalues():
			rex = r"[^\n,]+,[^,]+," + reject['model_year'] + ',[^,]+,[^,]+,[^,]+,(' + reject['car_name'] + r"),[^,]+,[^,]+[^,]+,[^,]+\n"
			x = re.match(rex, matches)
#check if non-match is already in file
#if not, then add it
			if not x:
				write_csv_line(of, add_na(reject, l_inst.keys()), keyset)


def step_two(filename):
	#dataset = read_dataset(filename)
	
	print "--------------------------------------------------"
	print str(filename)
	print "--------------------------------------------------"
	
	duplicates = check_for_duplicates(filename, ["model_year", "car_name"])
	inst_count = count_instances(filename)
	
	print "No. of instances: " + str(inst_count)
	print "Duplicates: " + str(duplicates)
	
def step_three(lfile,rfile,outfile):
	""" Joins two files into one, using a full outer join"""
	join_into_file( lfile, rfile, outfile)

def step_four(filein,fileout):
	""" Cleans data of intances that are missing mpg information. Prints new data
		information"""
	clean_data(filein,fileout)

	print "--------------------------------------------------"
	print "combined table (saved as " + str(filein) + " ):"
	print "--------------------------------------------------"

	inst_count = count_instances(filein)
	duplicates = check_for_duplicates(filein
		, ["model_year", "car_name"])
	
	print "No. of instances: " + str(inst_count)
	print "Duplicates: " + str(duplicates)
	
def step_five(filein):
	"""prints 5 number summaries for all relevant attributes"""
	print_summary_stats(filein)

def step_six(filein):
	""" Finds null values in each relevant attribute and replaces it with the 
		average for that attribute"""
	populate_empty_values(filein)

def main():

	step_two("auto-mpg.txt")
	step_two("auto-prices.txt")

	step_three("auto-mpg-nodups.txt","auto-prices-nodups.txt","auto-data.txt")

	#step_four("auto-data.txt","auto-data-cleaned.txt")

	#step_five("auto-data-cleaned.txt")

	step_six("auto-data-cleaned.txt")

	raw_input("Hit Enter to EXIT")

main()
