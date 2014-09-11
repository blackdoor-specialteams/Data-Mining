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
	with open(file_name, "r") as f:
		return sum(1 for _ in f)
	
def print_csv(table):
	"""Prints a table in csv format"""
	for row in table:
		for i in range(len(row)-1):
			sys.stdout.write(str(row[i])+ ',')
		print row[-1]

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

def print_summary_stats(file_name):
	print "Summary Stats:"
	print "============ ===== ===== ======= ====== ======"
	print " attribute    min   max    mid    avg    med"
	print "============ ===== ===== ======= ====== ======"
	
	summary_atts = [0,1,3,4,5,7,9]

	for x in summary_atts:
		att_list = get_att_list(file_name,x)
		statlist = summerize_data(att_list)
		print_att_summary_row(get_attribute_name(file_name,x),statlist)

def print_att_summary_row(attname,stat_list):
	print str(attname),
	for i in stat_list:
		print(x, '    ')
	print()

def get_att_list(filein,index):
	att_list = []
	with open(filein, 'rb') as _in:
		f1 = csv.reader(_in)
		for row in f1:
			if row[index] != 'NA':
				att_list.append(row[index])
	return att_list

def get_attribute_name(filein,index):
	with open(filein, 'rb') as _in:
		f1 = csv.reader(_in)
		line1 = f1.next()
	return line1[index]


def summerize_data(attlist):
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
        if (len(alist) % 2) == 0:  
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
	temp_file = 'soManyDups.tmp'
	match = True
	l_keyset = []
	r_keyset = []
	keyset = []
	rejects = dict()
	with open(out, 'wb') as of, open(l_file, 'r') as lf:
		l_reader = csv.DictReader(lf)
		for l_inst in l_reader:
			if not l_keyset: l_keyset = l_inst.keys()
			any_match = False
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
						any_match = True
						r_inst.update(l_inst)
						#print 'match\n'
						#print 'join' + str(r_inst) + '\n'
						write_csv_line(of, r_inst, keyset)#joined.append(r_inst)
					else:
						#write_csv_line(of, add_na(l_inst, r_inst.keys()), keyset)
						#write_csv_line(of, add_na(r_inst, l_inst.keys()), keyset)
						rejects[r_inst['model_year']+r_inst['car_name']] = r_inst
			if not any_match:
				write_csv_line(of, add_na(l_inst, r_inst.keys()), keyset)
		with open(out, 'rb') as matches:
			matches = matches.read()
		for reject in rejects.itervalues():
			rex = r"[^\n,]+,[^,]+," + reject['model_year'] + ',[^,]+,[^,]+,[^,]+,(' + reject['car_name'] + r"),[^,]+,[^,]+[^,]+,[^,]+\n"
			x = re.match(rex, matches)
			if not x:
				write_csv_line(of, add_na(reject, l_inst.keys()), keyset)


	#delete_dups_from_csv(temp_file,out)
	#os.remove(temp_file)

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
	
def step_three(lfile,rfile,outfile):
	join_into_file( lfile, rfile, outfile)

def step_four(filein,fileout):
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
	print_summary_stats(filein)

def main():

	step_two("auto-mpg.txt")
	step_two("auto-prices.txt")

	step_three("auto-mpg-nodups.txt","auto-prices-nodups.txt","auto-data.txt")

	#step_four("auto-data.txt","auto-data-cleaned.txt")

	#step_five("auto-data-cleaned.txt")

	raw_input("Hit Enter to EXIT")


main()
