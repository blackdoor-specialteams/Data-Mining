"""
@Names: Cj Buresch and Nathan Fischer
@Date: 9/15/2014
@ Homework #2 -- Data Visualization
@Description: 
	
@Version: Python v2.7
"""
#import matplotlib
#matplotlib.use('pdf')
import csv
import matplotlib.pyplot as plot
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
		plot.xticks(xrng,['foo','bar','baz','quz'])
		plot.yticks(yrng)
		# turn on the backround grid
		plot.grid(True)
		# save the result to a pdf file
		plot.savefig(str(attributes[att]) +'_freq.pdf')

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