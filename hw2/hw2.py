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
	yrng = numpy.arange(0,max(counts)+ 50,20)
	#create the bar chart
	pyplot.bar(xrng,counts,.45,align ='center')
	#define x and y ranges (and value labels)
	pyplot.xticks(xrng,values)
	pyplot.yticks(yrng)
	pyplot.xlabel(title)
	pyplot.ylabel("Count")

	# turn on the backround grid
	pyplot.grid(True)
	# save the result to a pdf file
	pyplot.savefig('step-1-' + title + '.pdf')
	pyplot.clf()

def create_cont_to_cat_graphs(table,index):
	run_approach_one(table,index)
	run_approach_two(table,index)

def run_approach_one(table,index):
	mpg_values = get_all_col_values(table,index)
	ratinglist = get_list_of_mpg_ratings(mpg_values)
	xs, ys = get_frequencies(ratinglist)
	xs = populate_rating_label_list(xs)
	graph_freq_diagram("Car MPG Rating",xs,ys)

def run_approach_two(table,index):
	#generate initial values
	mpg_values = get_all_col_values(table,index)
	ratinglist = get_list_of_mpg_ratings(mpg_values)
	#get graph cut off points from list
	cuttoffs = equal_width_cutoffs(ratinglist,5)
	bin_freq = frequencies_for_cutoffs(ratinglist,cuttoffs)

	graph_freq_diagram("Car MPG Rating by Range",cuttoffs,bin_freq)

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

def populate_rating_label_list(xs):
	labels = ["0-13","14","15-16","17-19","20-23","24-26","27-30","31-36","37-44",">45"]
	result = []
	for x in xs:
		if labels[x-1] not in result:
			result.append(labels[x-1])
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

def create_histogram(filein,exelist):
	return None

def create_scatter_plot(attributes,table,exelist):
	for x in exelist:
		graph_scatter_plot(attributes[x],table,x)

def graph_scatter_plot(attribute,table,index):	
	# reset figure
	pyplot.figure()
	pyplot.title(attribute + " vs MPG Scatter Plot")
	# create xs and ys
	xs , ys = get_points_from_table(table,index,1)
	# create the dot chart (with pcts)
	pyplot.plot(xs, ys, 'b.')
	# make axis a bit longer and wider
	pyplot.xlim(0, int(max(xs) * 1.10))
	pyplot.ylim(0, int(max(ys) * 1.10))
	pyplot.xlabel(attribute)
	pyplot.ylabel("MPG")

	pyplot.grid(True)
	pyplot.savefig('step-6-' +attribute + '.pdf')
	pyplot.clf()

def get_points_from_table(table,xindex,yindex):
	xresult = []
	yresult = []

	for row in table:
		if (row[xindex] != 'NA') and (row[yindex] != 'NA'):
			xresult.append(float(row[xindex]))
			yresult.append(float(row[yindex]))

	return xresult,yresult

def calculate_linear_regressions(filein,exelist):
	return None

def scatter_plot_with_regression(filein,exelist):
	return None

def run_step_8(attributes,table):
	create_box_plot(table)
	create_multi_freq_diagram(table)

def create_multi_freq_diagram(table):
	pyplot.figure()
	fig, ax = pyplot.subplots()
	
	years_list = get_all_model_years(table)
	#group data by year and get Origin info for that year
	grouped_by_year_list = group_By(table,2,8)
	list_of_freqs_by_year = []

	for xs in grouped_by_year_list:
		values , counts = get_frequencies(xs)
		list_of_freqs_by_year.append((values,counts))
	#Graph Settings 
	count_max = 0
	bar_width = 0.3
	locations = range(1,11)

	plot_counts = get_counts_for_origin_by_year(list_of_freqs_by_year,0)
	count_max = check_max(plot_counts,count_max)
	r1 = ax.bar(locations, plot_counts, bar_width, color='b')
	
	locations = increment_values_list(locations,bar_width)
	plot_counts = get_counts_for_origin_by_year(list_of_freqs_by_year,1)
	count_max = check_max(plot_counts,count_max)
	r2 = ax.bar(locations, plot_counts, bar_width, color='r')
	
	locations = increment_values_list(locations,bar_width)
	plot_counts = get_counts_for_origin_by_year(list_of_freqs_by_year,2)
	count_max = check_max(plot_counts,count_max)
	r3 = ax.bar(locations, plot_counts, bar_width, color='y')
	
	xrng = numpy.arange(1.5,len(list_of_freqs_by_year)+1)
	yrng = numpy.arange(0,count_max+ 25,5)

	ax.legend((r1[0], r2[0],r3[0]), ('USA', 'Europe','Japan'), loc=2)
	pyplot.xticks(xrng,years_list)
	pyplot.yticks(yrng)
	pyplot.xlabel("Model Year")
	pyplot.ylabel("Count")
	pyplot.title("Total Number of Cars by Year and Country of Origin")
	pyplot.grid(True)

	pyplot.savefig('step-8-Multiple-Freq.pdf')
	pyplot.clf()

def get_counts_for_origin_by_year(xs,index):
	result = []
	for x in xs:
		values = x[0]
		counts = x[1]
		result.append(counts[index])
	return result

def increment_values_list(xs,inc):
	result = []
	for x in xs:
		result.append(x + inc)
	return result

def check_max(xs,cmax):
	if max(xs) > cmax:
		return max(xs)
	else:
		return cmax                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                    

def create_box_plot(table):
	pyplot.figure()
	pyplot.title("MPG by Model Year")

	xs = group_By(table,2,1)

	pyplot.boxplot(xs)
	# set x-axis value names
	labels = get_all_model_years(table)
	pyplot.xticks(range(1,len(labels) + 1),labels)

	pyplot.xlabel("Model Year")
	pyplot.ylabel("MPG")
	pyplot.savefig('step-8-BoxPlot.pdf')
	pyplot.clf()

def get_all_col_values(table,index):
	result = []
	for row in table:
		if row[index] != "NA":
			result.append(int(float(row[index])))
	return result

def get_all_model_years(table):
	result = []
	for row in table:
		if row[2] != "NA":
				if int(float(row[2])) not in result:
					result.append(int(float(row[2])))
	result.sort()
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

def group_By(table,index,infoindex):
	group_vals = []
	for row in table:
		value = int(float(row[index]))
		if value not in group_vals:
			group_vals.append(value)
	group_vals.sort()

	result = [[] for _ in range(len(group_vals))]

	for row in table:
		result[group_vals.index(int(float(row[index])))].append(int(float(row[infoindex])))
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

def equal_width_cutoffs(values, num_of_bins):
	# find the approximate width
	width = int(max(values) - min(values)) / num_of_bins
	return list(range(min(values) + width, max(values), width))

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
	#create_cont_to_cat_graphs(datatable,1)

	scatterlist = [0,4,5,7,9]
	create_scatter_plot(attributes,datatable,scatterlist)

	run_step_8(attributes,datatable)


main()