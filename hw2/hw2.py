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
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as pyplot
import numpy
import math

def create_freq_diagram(filein,attlist):
	"""Create a Frequency diagram for each catagorical attributeute in the prepared dataset"""
	pyplot.figure()
	
	att_names = get_att_names(filein)

	for att in attlist:
		#set graph title form attributeute names
		pyplot.title(att_names[att])
		#define x and y values
		xs = []
		ys = []

		#get a list of all of the values from a column
		#get the count of frequienes from that list, for each value
		xs, ys = get_frequencies(get_all_att_values(filein,att))

		#calculate a range (make y bigger)
		xrng = numpy.arange(len(xs))
		yrng = numpy.arange(0,max(ys)+ 100,50)
		#create the bar chart
		pyplot.bar(xs,ys,.45,align ='center')
		#define x and y ranges (and value labels)
		pyplot.xticks(xrng,['foo','bar','baz','quz'])
		pyplot.yticks(yrng)
		# turn on the backround grid
		pyplot.grid(True)
		# save the result to a pdf file
		pyplot.savefig('fig1.pdf')
		pyplot.clf()

def graph_freq_diagram(table):
	return None

def create_pie_charts(filein, attlist = ['model_year', 'cylinders', 'origin']):
	for attribute in attlist:
		graph_pie_chart(filein, attribute, 'step-2-pie-'+ attribute +'.pdf', title = attribute + ' Pie Chart')
	return None

def graph_pie_chart(file_in, attribute, file_out, title = None):
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
	if title == None:
		title = attribute
	pyplot.title(title)
	pyplot.pie(data, labels = labels, autopct='%1.1f%%', colors=('b', 'g', 'r', 'c', 'm', 'y', 'w'))
	pyplot.savefig(file_out)
	pyplot.clf()

def create_dot_plots(file_in, attlist = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration', 'msrp']):
	for attribute in attlist:
		graph_dot_plot(file_in, attribute,  title = attribute + ' Dot Plot')
		save_fig('step-3-dot-'+ attribute +'.pdf')
	return None

def graph_dot_plot(file_in, attribute, title = None):
	pyplot.clf()
	pyplot.gca().get_yaxis().set_visible(False)
	with open(file_in, 'r') as f:
		reader = csv.DictReader(f)
		for inst in reader:
			pyplot.plot(float(inst[attribute]), 1, 'k.', alpha=.05, markersize=15)
	if title == None:
		title = attribute
	pyplot.title(title)
	pyplot.xlabel(attribute)
	
def create_histograms(file_in, attlist = ['mpg', 'displacement', 'horsepower', 'weight', 'acceleration', 'msrp']):
	for attribute in attlist:
		graph_histogram(file_in, attribute,  title = attribute + ' Histogram')
		save_fig('step-5-histo-'+ attribute +'.pdf')
	return None

def graph_histogram(file_in, attribute, title = None):
	pyplot.clf()
	if title == None:
		title = attribute
	xs = []
	with open(file_in, 'r') as f:
		reader = csv.DictReader(f)
		for inst in reader:
			xs.append(float(inst[attribute]))
	pyplot.title(title)
	pyplot.xlabel(attribute)
	pyplot.ylabel('Instances')
	pyplot.hist(xs)

def save_fig(file_out):
	pyplot.savefig(file_out)
	pyplot.clf()

def create_scatter_plot(filein,attlist):
	return None

#returns a three-tuple for point slope form (x, y, slope)
def calculate_best_fit_line(xs, ys):
	if len(xs) != len(ys):
		return None
	_x = numpy.mean(xs)
	_y = numpy.mean(ys)
	num = 0
	denom = 0
	for x_point, y_point in zip(xs, ys):
		num += (x_point-_x)*(y_point-_y)
		denom += math.pow(x_point-_x, 2)
	m = num/denom
	return (_x, _y, m)

def calculate_correlation_coefficient(xs, ys, m):
	return m * numpy.std(xs) / numpy.std(ys)

def calculate_covariance(m, xs, sig_x = None, _x = None):
	#return m * math.pow(numpy.std(xs), 2)
	return m * math.pow(sig_x if sig_x!=None else numpy.std(xs), 2) * 1.0

def graph_line(x, y, m, min_x, max_x):
	xs = [x1 for x1 in range(min_x, max_x)]
	ys = [-1*m*(x-x1)+y for x1 in range(min_x, max_x)]
	pyplot.plot(xs, ys)

def create_linear_regressions_with_scatters(file_in, x_attribs = ['displacement', 'horsepower', 'weight', 'msrp', 'displacement'], y_attribs = ['mpg']*4+['weight']):
	for x_attrib, y_attrib in zip(x_attribs, y_attribs):
		graph_scatter_plot_with_regression(file_in, x_attrib, y_attrib)
		save_fig('step-7-'+ x_attrib +'-vs-'+ y_attrib +'.pdf')
	return None

def graph_scatter_plot_with_regression(file_in, x_attrib, y_attrib, title = None):
	pyplot.clf()
	xs = []
	ys = []
	with open(file_in, 'r') as f:
		reader = csv.DictReader(f)
		for inst in reader:
			xs.append(float(inst[x_attrib]))
			ys.append(float(inst[y_attrib]))
	ys = sorted(ys)
	xs = sorted(xs)
	x, y, m = calculate_best_fit_line(xs, ys)
	graph_line(x, y, m, xs[0], xs[-1])
	b = dict(facecolor='none', color='r')
	s = 'r=' + str(calculate_correlation_coefficient(xs, ys, m))[0:8] + ' cov=' + str(calculate_covariance(m, xs))[0:8]
	pyplot.text(xs[0], ys[0], s, bbox=b, fontsize=10, color='r')
	pyplot.title(title if title!=None else str(x_attrib) + ' vs ' + str(y_attrib))
	pyplot.xlabel(x_attrib)
	pyplot.ylabel(y_attrib)
	pyplot.plot(xs, ys, '.')
	return None

def get_att_names(filein):
	with open(filein, 'rb') as _in:
		f1 = csv.reader(_in)
		names = f1.next()
		return names

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
		f1 = csv.reader(_in)
		next(f1)
		for row in f1:
			if row[index] != 'NA':
				att_list.append(float(row[index]))
	return att_list

def main():
	catagorical_att_list = [2,3,8]
	data = "auto-data-cleaned.txt"

	create_freq_diagram(data,catagorical_att_list)

main()