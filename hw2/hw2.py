"""
@Names: Cj Buresch and Nathan Fischer
@Date: 9/15/2014
@ Homework #2 -- Data Visualization
@Description: 
	
@Version: Python v2.7
"""
import matplotlib
matplotlib.use('pdf')
import matplotlib.pyplot as pyplot

def create_freq_diagram(filein,attlist):
	"""Create a Frequency diagram for each catagorical attribute in the prepared dataset"""
	pyplot.figure()
	#define x and y values
	xs = []
	ys = []
	#calculate a range (make y bigger)
	xrng = numpy.arange(len(xs))
	yrng = numpy.arange(0,max(ys)+ 100,50)
	#create the bar chart
	pyplot.bar(xrng,ys,.45,align ='center')
	#define x and y ranges (and value labels)
	pyplot.xticks(xrng,['foo','bar','baz','quz'])
	pylot.yticks(yrng)
	# turn on the backround grid
	pyplot.grid(True)
	# save the result to a pdf file
	pyplot.savefig('fig1.pdf')

def create_pie_chart(filein,attlist):

def create_a_dot_plot(filein,attlist):

def create_histogram(filein,attlist):

def create_scatter_plot(filein,attlist):

def calculate_linear_regressions(filein,attlist):

def scatter_plot_with_regression(filein,attlist):

def get_get_attName():

def get_frequencies(xs):
	ys = sorted(xs)
	values, counts = [], []
	for y in ys:
		if y not in values:
			values.append(y)
			count.append(1)
		else:
			count[-1] += 1
		return values, counts

def main():
	catagorical_att_list = [2,3,8]
	data = "auto-data-cleaned.txt"

	create_freq_diagram(data,catagorical_att_list)

main()