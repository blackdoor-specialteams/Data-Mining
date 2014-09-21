"""
@Names: Cj Buresch and Nathan Fischer
@Date: 9/15/2014
@ Homework #2 -- Data Visualization
@Description: 
	
@Version: Python v2.7
"""

import csv
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as pyplot
import numpy

def create_freq_diagram(attributes,table,exelist):
	"""Create a Frequency diagram for each catagorical attribute in the prepared dataset"""
	for att in exelist:
		#define x and y values
		xs = []
		ys = []

		#get a list of all of the values from a column
		#get the count of frequienes from that list, for each value
		xs, ys = get_frequencies(get_all_col_values(table,att))
		graph_freq_diagram(attributes[att],xs,ys)

def graph_freq_diagram(title,values,counts):
	pyplot.figure()
	pyplot.title(title + " Frequency Diagram")
	# calculate a range (make y a bit bigger)
	xrng = numpy.arange(len(values))
	yrng = numpy.arange(0,max(counts)+ 100,50)
	#create the bar chart
	pyplot.bar(xrng,counts,.45,align ='center')
	#define x and y ranges (and value labels)
	pyplot.xticks(xrng,values)
	pyplot.yticks(yrng)

	# turn on the backround grid
	pyplot.grid(True)
	# save the result to a pdf file
	pyplot.savefig(title + '_freq.pdf')
	pyplot.clf()

def create_cont_to_cat_graphs(table,index):
	run_approach_one(table,index)

def run_approach_one(table,index):
	mpg_values = get_all_col_values(table,index)
	ratinglist = get_list_of_mpg_ratings(mpg_values)
	xs, ys = get_frequencies(ratinglist)
	xs = ["0-13","14","15-16","17-19","20-23","24-26","27-30","31-36","37-44",">45"]
	graph_freq_diagram("Car MPG Rating Frequency Diagram",xs,ys)

def get_list_of_mpg_ratings(xs):
	result = []
	for x in xs:
		if x >= 45:
			result.append(10)
		elif x >= 37:
			result.append(9)
		elif x >= 31:
			result.append(8)
		elif x >= 27:
			result.append(7)
		elif x >= 24:
			result.append(6)
		elif x >= 20:
			result.append(5)
		elif x >= 17:
			result.append(4)
		elif x >= 15:
			result.append(3)
		elif x == 14:
			result.append(2)
		elif x <= 13:
			result.append(1)
	return result

def create_pie_charts(filein, attlist = ['Model Year', 'Cylinders', 'Origin']):
	for attribute in attlist:
		graph_pie_chart(filein, attribute, 'step-2-pie-'+ attribute +'.pdf')
	return None

def graph_pie_chart(file_in, attribute, file_out):
	labels = []
	data = []
	with open(file_in, 'r') as f:
		reader = csv.DictReader(f)
		for inst in reader:
			if str(inst[attribute]) in labels:
				data[labels.index(str(inst[attribute]))] += 1
			else:
				labels.append(str(inst[attribute]))
				data.append(1)
	pyplot.title(attribute)
	pyplot.pie(data, labels = labels, autopct='%1.1f%%', colors=('b', 'g', 'r', 'c', 'm', 'y', 'w'))
	pyplot.savefig(file_out)
	pyplot.clf()

def create_a_dot_plot(filein,attlist):
	return None

def create_pie_chart(filein,exelist):
	return None

def create_a_dot_plot(filein,exelist):
	return None

def create_histogram(filein,exelist):
	return None

def create_scatter_plot(filein,exelist):
	return None

def calculate_linear_regressions(filein,exelist):
	return None

def scatter_plot_with_regression(filein,exelist):
	return None

def get_all_col_values(table,index):
	result = []
	for row in table:
		if row[index] != "NA":
			result.append(float(row[index]))
	return result

def get_frequencies(xs):
	ys = sorted(xs)
	values, counts = [], []
	for y in ys:
		if y not in values:
			values.append(y)
			counts.append(1)
		else:
			counts[-1] += 1
	return values, counts

def group_By(table,index):
	group_vals = []
	for row in table:
		value = row[index]
		if value not in group_vals:
			group_vals.append(value)
	group_vals.sort()

	# create list of n empty partitions
	result = [[] for _ in range(len(group_vals))]

	for row in table:
		result[group_vals.index(row[index])].append(row[:])

	return result

def frequencies_for_cutoffs(values, cutoff_points):
	new_values = []
	for v in values:
		found = False
		for i in range(len(cutoff_points)):
			if not found and v <= cutoff_points[i]:
				new_values.append(i + 1)
		if not found:
			new_values.append(len(cutoff_points) + 1)
	return new_values

def get_table_from_CSV(filename):
	table =[]
	atts = []
	with open(filename, 'rb') as _in:
		f1 = csv.reader(_in)
		atts = f1.next()
		for row in f1:
			if len(row) > 0:
				table.append(row)
	return atts, table


def print_table(table):
	for row in table:
		print row

def main():
	#Relevant Attributes to graph
	catagorical_att_list = [2,3,8]
	inputdata = "auto-data-cleaned.txt"
	#Get list of attributes, and the table from the input data
	attributes,datatable = get_table_from_CSV(inputdata)

	create_freq_diagram(attributes,datatable,catagorical_att_list)
	create_cont_to_cat_graphs(datatable,1)


main()