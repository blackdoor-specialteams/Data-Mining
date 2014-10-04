from cj import *
import hw2

#n: 1,2,6
#CJ: 3,4,5

''' Create a classifier that predicts mpg values using a (least squares) linear regression based on vehicle
weight. Your classifier should take one or more instances, compute the predicted MPG values, and then
map these to the DOE classification/ranking (given in HW 2) for each corresponding instance. Test your
classifier by selecting 5 random instances (via your script) from the dataset, predict their corresponding mpg
ranking, and then show their actual mpg ranking as follows:
===========================================
STEP 1: Linear Regression MPG Classifier
===========================================
instance: 15.0, 8, 400.0, 150.0, 3761, 9.5, 70, 1, chevrolet monte carlo, 3123
class: 4, actual: 3
instance: 17.0, 8, 302.0, 140.0, 3449, 10.5, 70, 1, ford torino, 2778
class: 5, actual: 4
instance: 28.4, 4, 151.0, 90.00, 2670, 16.0, 79, 1, buick skylark limited, 4462
class: 6, actual: 7
instance: 20.0, 6, 232.0, 100.0, 2914, 16.0, 75, 1, amc gremlin, 2798
class: 5, actual: 5
instance: 16.2, 6, 163.0, 133.0, 3410, 15.8, 78, 2, peugeot 604sl, 10990
class: 5, actual: 4
Note you should run your program enough times to check that the approach is working correctly'''
def step1():
	return None

'''Create a k = 5 nearest neighbor classifier for mpg that uses the number of cylinders, weight, and
acceleration attributes to predict mpg. Be sure to normalize the MPG values and also use the Euclidean
distance metric. Similar to Step 1, test your classifier by selecting random instances from the dataset, predict
their corresponding mpg ranking, and then show their actual mpg ranking:
===========================================
STEP 2: k=5 Nearest Neighbor MPG Classifier
===========================================
instance: 15.0, 8, 400.0, 150.0, 3761, 9.5, 70, 1, chevrolet monte carlo, 3123
class: 7, actual: 3
instance: 17.0, 8, 302.0, 140.0, 3449, 10.5, 70, 1, ford torino, 2778
class: 7, actual: 4
instance: 28.4, 4, 151.0, 90.00, 2670, 16.0, 79, 1, buick skylark limited, 4462
class: 1, actual: 7
1instance: 20.0, 6, 232.0, 100.0, 2914, 16.0, 75, 1, amc gremlin, 2798
class: 1, actual: 5
instance: 16.2, 6, 163.0, 133.0, 3410, 15.8, 78, 2, peugeot 604sl, 10990
class: 7, actual: 4
'''
def step2():
	return None

'''Create two versions of a Na¨ıve Bayes classifier to predict mpg based on the number of cylinders,
weight, and model year attributes. For the first one, create a categorical version of weight using the following
classifications (based on NHTSA vehicle sizes).
Ranking Range
5 ≥ 3500
4 3000–3499
3 2500–2999
2 2000–2499
1 ≤ 1999
For the second, calculate the conditional probability for weight using the Gaussian distribution function from
class. Similar to Step 1, test your classifier by selecting random instances from the dataset, predict their
corresponding mpg ranking, and then show their actual mpg ranking:
'''
def step3():
	return None

'''Compute the predictive accuracy (and standard error) of the four classifiers using separate training
and test sets. You should use two approaches for testing. The first approach should use random subsampling
with k = 10. The second approach should use stratified k-fold cross validation with k = 10. Your output
should look something like this (where the ??’s should be replaced by actual values):
===========================================
STEP 4: Predictive Accuracy
===========================================
Random Subsample (k=10, 2:1 Train/Test)
Linear Regression: p = 0.?? +- 0.??
2Naive Bayes I: p = 0.?? +- 0.??
Naive Bayes II: p = 0.?? +- 0.??
Top-K Nearest Neighbor: p = 0.?? +- 0.??
Stratified 10-Fold Cross Validation
Linear Regression: p = 0.?? +- 0.??
Naive Bayes I: p = 0.?? +- 0.??
Naive Bayes II: p = 0.?? +- 0.??
Top-K Nearest Neighbor: p = 0.?? +- 0.??
'''
def step4():
	return None

'''Create confusion matrices for each classifier. You can use the tabulate package to display your
confusion matrices (it is also okay to format the table manually). Here is an example:
Linear Regression (Stratified 10-Fold Cross Validation):
===== === === === === === === === === === ==== ======= =================
MPG 	1 	2 	3 	4 	5 	6 	7 	8 	9 	10 	Total 	Recognition (%)
===== === === === === === === === === === ==== ======= =================
	1 	14 	2 	5 	3 	1 	0 	0 	3 	0 	0 	25 		56
2 5 3 6 1 1 0 0 0 0 0 16 18.75
3 3 5 9 8 6 0 0 0 0 0 31 29.03
4 0 2 4 18 21 2 3 0 0 0 50 36
5 0 0 0 6 27 15 3 0 0 0 51 52.94
6 0 0 0 1 3 12 15 0 0 0 31 38.71
7 0 0 0 0 1 6 19 0 0 0 26 73.08
8 0 0 0 0 0 1 18 0 1 0 20 0
9 0 0 0 0 0 0 3 0 0 0 3 0
10 0 0 0 0 0 0 0 0 0 0 0 0
===== === === === === === === === === === ==== ======= =================
Naive Bayes I (Stratified 10-Fold Cross Validation):
===== === === === === === === === === === ==== ======= =================
MPG 1 2 3 4 5 6 7 8 9 10 Total Recognition (%)
===== === === === === === === === === === ==== ======= =================
1 20 4 1 0 0 0 0 0 0 0 25 80
2 6 8 2 0 0 0 0 0 0 0 16 50
3 7 6 9 7 2 0 0 0 0 0 31 29.03
4 3 1 7 27 10 2 0 0 0 0 50 54
5 0 0 1 18 22 9 1 0 0 0 51 43.14
6 0 0 0 2 6 17 3 3 0 0 31 54.84
7 0 0 0 0 5 7 11 3 0 0 26 42.31
8 0 0 0 0 1 3 3 13 0 0 20 65
9 0 0 0 0 0 0 0 3 0 0 3 0
10 0 0 0 0 0 0 0 0 0 0 0 0
===== === === === === === === === === === ==== ======= =================
...
'''
def step5():
	return None

''' Use Na¨ıve Bayes and k-nearest neighbor to create two different classifiers to predict survival from the
titanic dataset (titanic.txt). Note that the first line of the dataset lists the name of each attribute (class,
age, sex, and surivived). Your classifiers should use class, age, and sex attributes to determine the survival
class. Be sure to write down any assumptions you make in creating the classifiers. Evaluate the performance
of your classifier using stratified k-fold cross validation (with k = 10) and generate confusion matrices for
the two classifiers.'''
def step6():
	return None


def main():
	my_function()

main()