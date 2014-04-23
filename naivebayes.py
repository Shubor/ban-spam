import csv
import math

N = 200
FOLD = 10

#======|| Read in csv files ||======#

w_body_legit, w_body_spam = [], []
w_subj_legit, w_subj_spam = [], []

# Read CSV:
#	Read in given csv file
#	Saves data from file into either legit or spam lists
#		depending on class of file given by last column of csv
def read_csv( file_name, legit, spam ):

	with open( file_name, "rb" ) as f:
		row_num = 0
		for row in csv.reader(f):
			# Ignore header
			if row_num != 0:
				# Check class of file (stored in last row of csv)
				if row[-1] == "nonspam":
					# Append row, removing class of doc
					legit.append( row[:-1] )
				else:
					spam.append( row[:-1] )
			row_num += 1

read_csv( "body.csv", w_body_legit, w_body_spam )
read_csv( "subject.csv", w_subj_legit, w_subj_spam )

#############################################################################
# Calculate Naive Bayes, assuming normal distribution                       #
#############################################################################

#======|| Calculate Mean ||======#

# Mean:
#	Calculates the average value of a set
# 	returns float
def mean( column ):

	return math.fsum( [float(row) for row in column] ) / len( column )

#======|| Calculate Standard Deviation ||======#

# Standard dev:
#	Calculates the standard deviation of a set
# 	returns: float
def std_dev( column ):

	u = mean(column)
	return math.sqrt( math.fsum([((float(x) - u) ** 2) for x in column]) / ( len(column) - 1 ) )

#======|| Probability Density Function ||======#

# Probability density function
#	Calculates probability of x assuming normal distribution
#	x = value; u = mean; s = standard deviation
# 	returns float
def prob_density( x, u, s ):

	if s==0 and x==u:
		return 1
	elif s==0 and x!=u:
		return 0

	coefficient = ( s * math.sqrt( 2 * math.pi ) )
	exponent = - ((x - u) ** 2) / (2 * (s ** 2))

	return coefficient * math.exp(exponent)

#=========|| Naive Bayes ||=========#
# Given: a vector of cosine normed tdidf values of top200 document set terms
#
# Find:	document frequency of top200 terms
#	vector of cosine normed tdidf values of top200 document set terms
# Method: apply naive bayes using above vector for spam and nonspam
#	  choose "nonspam" for ties
# Return: "spam" or "nonspam"

# Split:
#	Split data into even chunks
def split( data ):

	n = len( data ) / float( FOLD )
	out = []
	last = 0.0
	while last < len( data ):
		out.append( data[ int(last) : int(last + n) ] )
		last += n
	return out

# Train:
#	Calculate mean and standard deviation of data based on given training data
def train( train_legit, train_spam ):

	mean_legit, mean_spam = [], []
	sd_legit, sd_spam = [], []

	# Calculate mean and standard deviation of column fn
	for n in range( len( train_legit[0] ) ):
		mean_legit.append( mean(  [row[n] for row in train_legit] ) )
		sd_legit.append( std_dev( [row[n] for row in train_legit] ) )

	for n in range( len( train_spam[0] ) ):
		mean_spam.append( mean(  [row[n] for row in train_spam] ) )
		sd_spam.append( std_dev( [row[n] for row in train_spam] ) )

	return mean_legit, mean_spam, sd_legit, sd_spam

# Classify:
# 	Get P(spam|document) = P(f1|spam)*.....*P(f200|spam)
# 	Similarly, calculate P(nonspam|document)
#	Compare P(spam|document) and P(nonspam|document)
#	If there is a tie, choose non-spam
#		Should implement a threshold for this to avoid false positives
def classify( test_legit, test_spam, mean_legit, mean_spam, sd_legit, sd_spam ):

	# Number of examples = 60, prob_spam = number of examples which are spam / examples = 20/60
	TOTAL_DOCS = float( len(test_spam) + len(test_legit) ) # Number of examples = 60
	P_spam  = len( test_spam )  / TOTAL_DOCS	# P(SPAM) is |SPAM|/|EXAMPLES| = 0.33334
	P_legit = len( test_legit ) / TOTAL_DOCS # P(NONSPAM) is 1-P(SPAM) = 0.6666667
	SPAM = "spam"
	LEGIT = "nonspam"
	num_correct = 0

	Lp = 0.05 # Laplace correction used by weka
	C = 2.0 # Threshold for spam: P(spam|X=x) > C P(C=legit|X=x)

	# Test on the known legitimate documents
	for document in test_legit:

		P_legit_X, P_spam_X = 1.0, 1.0

		# Calculate P(class|X)=P(X1|class) ... P(X200|class) P(class)
		for x in range(len(document)):
			P_spam_X  *= prob_density( float(document[x]), mean_spam[x] , sd_spam[x]  )
			P_legit_X *= prob_density( float(document[x]), mean_legit[x], sd_legit[x] )

		# Laplace correction because P(Xn|class) = 0 => P(class|X)=0 
		if P_legit_X==0 or P_spam_X==0:
			P_legit_X, P_spam_X = 1.0, 1.0			
			for x in range(len(document)):
				p_s = prob_density( float(document[x]), mean_spam[x], sd_spam[x] )
				p_l = prob_density( float(document[x]), mean_legit[x], sd_legit[x] )
	
				# P(X=xn|class) -> Lp if otherwise was P(X=xn|class)=0
				if p_s == 0:
					P_spam_X *= Lp
				else:
					P_spam_X *= p_s
				
				if p_l == 0:
					P_legit_X *= Lp
				else:
					P_legit_X *= p_l

		# P(class|X) = P(X|class) P(class)
		P_legit_X *= P_legit
		P_spam_X  *= P_spam

		# Classify
		if C * P_legit_X >= P_spam_X:
			num_correct += 1

	# Test on the known spam documents
	for document in test_spam:
		P_legit_X, P_spam_X = 1.0, 1.0

		# Calculate P(class|X)=P(X1|class) ... P(X200|class) P(class)
		for x in range(len(document)):
			P_spam_X  *= prob_density( float(document[x]), mean_spam[x] , sd_spam[x]  )
			P_legit_X *= prob_density( float(document[x]), mean_legit[x], sd_legit[x] )

		# Laplace correction because P(Xn|class) = 0 => P(class|X)=0 
		if P_legit_X==0 or P_spam_X==0:
			P_legit_X, P_spam_X = 1.0, 1.0			
			for x in range(len(document)):
				p_s = prob_density( float(document[x]), mean_spam[x], sd_spam[x] )
				p_l = prob_density( float(document[x]), mean_legit[x], sd_legit[x] )
	
				# P(X=xn|class) -> Lp if otherwise was P(X=xn|class)=0
				if p_s == 0:
					P_spam_X *= Lp
				else:
					P_spam_X *= p_s
				
				if p_l == 0:
					P_legit_X *= Lp
				else:
					P_legit_X *= p_l	

		# P(class|X) = P(X|class) P(class)
		P_legit_X *= P_legit
		P_spam_X  *= P_spam

		# Classify
		if C * P_legit_X < P_spam_X:
			num_correct += 1

	# Accuracy
	return num_correct / TOTAL_DOCS

#=========|| 10-FOLD stratified cross validation ||=========#

# Split each set into ten parts, 40(/20) nonspam(/spam), in order
sp_body_legit = split( w_body_legit )
sp_body_spam  = split( w_body_spam  )
sp_subj_legit = split( w_subj_legit )
sp_subj_spam  = split( w_subj_spam  )

# Save examples in each fold to csv file
with open( "body-folds.csv", "wb" ) as f:
	writer = csv.writer(f)

	for n in range( FOLD ):
		writer.writerow( [ "fold" + str( n + 1 ) ] )
		writer.writerows( [row + ["nonspam"] for row in sp_body_legit[n] ] )
		writer.writerows( [row + ["spam"] for row in sp_body_spam[n] ] )
		writer.writerow( [ ] ) # Empty line
	f.close()

sum_accuracy = 0.0

#========|| Perform over K-groups ||========#

# Iterate for each for fold
for test_num in range(0,10):
	mean_legit, mean_spam = [], []
	sd_legit, sd_spam = [], []

	# Create training data from other 9/10 of the examples.
	train_legit = []
	train_spam = []

	for training_num in range( 0, FOLD ):
		if training_num != test_num:
			train_legit.extend( sp_body_legit[training_num] )
			train_spam.extend(  sp_body_spam[training_num]  )

	# Train data
	# mean_legit[9] is the mean of 1:360 => test_legit is 361:400
	mean_legit, mean_spam, sd_legit, sd_spam = train( train_legit, train_spam )

	# sp_body_legit[9] is 361:400
	# works for mean_legit and sd_legit

	# Checking accuracy on test data 361:400
	sum_accuracy += classify( sp_body_legit[test_num], sp_body_spam[test_num], mean_legit, mean_spam, sd_legit, sd_spam )

print sum_accuracy / 10.0


