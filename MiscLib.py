def cfor(first,test,update):
	while test(first):
		yield first
		first = update(first)

def example(blah):
	print "do some stuff"
	for i in cfor(0,lambda i:i<blah,lambda i:i+1):
		print i
	print "done"
