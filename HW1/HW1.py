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
import os

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

def print_summary_stats(file_name):
	print "Summary Stats:"
	print "============ ===== ===== ======= ====== ======"
	print " attribute    min   max    mid    avg    med"
	print "============ ===== ===== ======= ====== ======"

def delete_dups_from_csv(filename_in,filename_out):
	""" Takes a CSV file, writes a second file that omits all duplicates"""
	f1 = csv.reader(open(filename_in, 'rb'))
	writer = csv.writer(open(filename_out, "wb"))
	
	unique_rows = []
	for row in f1:
		if row not in unique_rows:
			writer.writerow(row)
			unique_rows.append(row)

def attribute_stats(attlist):
	statlist = []

	lmin = min(attlist)
	lmax = max(attlist)
	lmid = midpoint(lmin,lmax)
	lavg = avg_list(attlist)
	lmed = median(attlist)

	statlist.append(lmin)
	statlist.append(lmax)
	statlist.append(lmid)
	statlist.append(lavg)
	statlist.append(lmed)
	return statlist

def median(alist):
        srtd = sorted(alist) 
        mid = len(alist)/2   
        if len(alist) % 2 == 0:  
                return (srtd[mid-1] + srtd[mid]) / 2.0
        else:
                return srtd[mid]

def average(numbers):  
        return float(sum(numbers))/len(numbers)

def midpoint(a,b):  
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

def write_csv_line(fileinst, d, keyset):
	line = ''
	for key in keyset:
		line += str(d[key]) + ','
	fileinst.write(line[0:-1] + '\n')

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
	l_keyset = []
	r_keyset = []
	keyset = []
	with open(out, 'wb') as of:
		with open(l_file, 'r') as lf:
			l_reader = csv.DictReader(lf)
			for l_inst in l_reader:
				if not l_keyset: l_keyset = l_inst.keys()
				#print l_inst
				with open(r_file, 'r') as rf:
					r_reader = csv.DictReader(rf)
					for r_inst in r_reader:
						if not r_keyset: 
							r_keyset = r_inst.keys()
							keyset += l_keyset
							for key in r_keyset:
								if key not in keyset:
									keyset.append(key)
							write_csv_header(of, keyset)
						#print r_inst
						match = True
						for key in keys:
							match &= l_inst[key] == r_inst[key]
						if match:
							r_inst.update(l_inst)
							#print 'match\n'
							#print 'join' + str(r_inst) + '\n'
							write_csv_line(of, r_inst, keyset)#joined.append(r_inst)
						else:
							#print 'nomatch\n'
							#print 'join' + str(l_inst) + '\n' #joined.append(l_inst)
							write_csv_line(of, add_na(l_inst, r_inst.keys()), keyset)
							write_csv_line(of, add_na(r_inst, l_inst.keys()), keyset)
							#print 'unmatched' + str(r_inst) + '\n'#r_unmatched[compound] = add_na(r_inst, l_inst.keys())


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
	step_two("auto-mpg-nodups.txt")
	step_two("auto-prices-nodups.txt")
	#delete_dups_from_csv()
	
	raw_input("Hit Enter to EXIT")


main()