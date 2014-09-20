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
	plot.figure()

	for att in exelist:
		#set graph title form attribute names
		plot.title(attributes[att])
		#define x and y values
		xs = []
		ys = []

		#get a list of all of the values from a column
		#get the count of frequienes from that list, for each value
		xs, ys = get_frequencies(get_all_col_values(table,att))

		#calculate a range (make y bigger)
		xrng = numpy.arange(len(xs))
		yrng = numpy.arange(0,max(ys)+ 100,50)
		#create the bar chart
		plot.bar(xs,ys,.45,align ='center')
		#define x and y ranges (and value labels)

		pyplot.xticks(xrng,['foo','bar','baz','quz'])
		pyplot.yticks(yrng)

		# turn on the backround grid
		plot.grid(True)
		# save the result to a pdf file

		pyplot.savefig('fig1.pdf')
		pyplot.clf()

def graph_freq_diagram(table):


def create_pie_charts(filein, attlist = ['model_year', 'cylinders', 'origin']):
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
		plot.savefig(str(attributes[att]) +'_freq.pdf')

def create_pie_chart(filein,exelist):
	return None

def create_a_dot_plot(filein,exelist):
>>>>>>> work working work
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
		if row[index] != 'NA':
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
			counts[y] += 1
		return values, counts


def get_all_att_values(filein,index):
	""" Collects all attributeute data from a csv file for a single attributeute"""
	att_list = []
	with open(filein, 'rb') as _in:

def group_By(inputTable):
	return None

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


def main():
	#Relevant Attributes to graph
	catagorical_att_list = [2,3,8]
	inputdata = "auto-data-cleaned.txt"
	#Get list of attributes, and the table from the input data
	attributes,dataset = get_table_from_CSV(inputdata)

	create_freq_diagram(attributes,dataset,catagorical_att_list)

main()