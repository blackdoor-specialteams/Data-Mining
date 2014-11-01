import math

"""
###################################################################################################
			Linear Regression Classifier
###################################################################################################
How to Use:
"""

def calculate_best_fit_line(xs, ys):
	if len(xs) != len(ys):
		return None
	_x = float(sum(xs)) / float(len(xs))
	_y = float(sum(ys)) / float(len(ys))
	num = 0
	denom = 0
	for x_point, y_point in zip(xs, ys):
		num += (x_point-_x)*(y_point-_y)
		denom += math.pow(x_point-_x, 2)
	m = num/denom
	return (_x, _y, m)

def lr_classify(_x, _y, m, x = None, y = None):
	if y == None and x == None:
		return None
	if(y == None): # solve for y
		return m * (x - _x) + _y /1.0
	else:# solve for x
		return (y -_y)/m + _x /1.0