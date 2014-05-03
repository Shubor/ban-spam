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

	with open( file_name, "r" ) as f:
		for row_num, row in enumerate( csv.reader(f) ):
			# Ignore header
			if row_num != 0:
				# Check class of file (stored in last column of csv)
				if row[-1] == "nonspam":
					# Append row, removing class of doc
					legit.append( row[:-1] )
				elif row[-1] == "spam":
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

def pdf( x, u, s ):

	if s == 0.0 and x == u:
		return HIGH_DENSITY

	elif s == 0.0 and x != u:
		return TINY_DENSITY

	coefficient = 1.0 / ( s * math.sqrt( 2.0 * math.pi ) )
	exponent = - ((x - u) ** 2.0) / (2.0 * (s ** 2.0))

	density = coefficient * math.exp(exponent)

	if density == 0:
		return LOW_DENSITY

	return density

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

	# 40 * 9 => legit emails
	# 20 * 9 => spam emails

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
#	Naive Bayes classifier
# 	Get P(spam|document) = P(f1|spam)*.....*P(f200|spam)
# 	Similarly, calculate P(nonspam|document)
#	Compare P(spam|document) and P(nonspam|document)
#	If there is a tie, choose non-spam
#		Should implement a threshold for this to avoid false positives
def classify( test_legit, test_spam, mean_legit, mean_spam, sd_legit, sd_spam ):

	TOTAL_DOCS = float( len(test_spam) + len(test_legit) ) # The size of the test set
	P_spam  = len( test_spam )  / TOTAL_DOCS # P(SPAM) is |SPAM|/|EXAMPLES| = 0.33334
	P_legit = len( test_legit ) / TOTAL_DOCS # P(NONSPAM) is 1-P(SPAM) = 0.6666667

	true_positive = 0 # Correctly classified as spam
	true_negative = 0 # Correctly classified as nonspam

	false_positive = 0 # Incorrectly classified as spam
	false_negative = 0 # Incorrectly classified as nonspam

	# Test on the known legitimate documents
	for document in test_legit:

		P_legit_X, P_spam_X = 0.0, 0.0

		# Calculate P(class|X)=P(X1|class) ... P(X200|class) P(class)
		for x in range(len(document)):

			P_X_spam  = pdf( float(document[x]), mean_spam[x] , sd_spam[x]  )
			P_X_legit = pdf( float(document[x]), mean_legit[x], sd_legit[x] )

			P_spam_X += math.log(P_X_spam)
			P_legit_X += math.log(P_X_legit)

		# P(class|X) = P(X|class) P(class)
		P_legit_X += math.log(P_legit)
		P_spam_X  += math.log(P_spam)

		# Classify
		if P_legit_X >= P_spam_X:
			true_negative += 1
		else:
			false_positive +=1

	# Test on the known spam documents
	for document in test_spam:
		P_legit_X, P_spam_X = 0.0, 0.0

		# Calculate P(class|X)=P(X1|class) ... P(X200|class) P(class)
		for x in range(len(document)):
			P_X_spam  = pdf( float(document[x]), mean_spam[x] , sd_spam[x]  )
			P_X_legit = pdf( float(document[x]), mean_legit[x], sd_legit[x] )

			P_spam_X += math.log(P_X_spam)
			P_legit_X += math.log(P_X_legit)

		# P(class|X) = P(X|class) P(class)
		P_legit_X += math.log(P_legit)
		P_spam_X  += math.log(P_spam)

		# Classify
		if P_legit_X < P_spam_X:
			true_positive += 1
		else:
			false_negative += 1

	# Accuracy
	return (true_negative + true_positive) / (true_negative + true_positive + false_positive + false_negative)

#=========|| 10-FOLD stratified cross validation ||=========#

# Split each set into ten parts, 40(/20) nonspam(/spam), in order
sp_body_legit = split( w_body_legit )
sp_body_spam  = split( w_body_spam  )
sp_subj_legit = split( w_subj_legit )
sp_subj_spam  = split( w_subj_spam  )

# Save examples in each fold to csv file
with open( "body-folds.csv", "w" ) as f:
	writer = csv.writer(f)

	for n in range( FOLD ):
		writer.writerow( [ "fold" +  str( n + 1 ) ] )
		writer.writerows( [row + ["nonspam"] for row in sp_body_legit[n] ] )
		writer.writerows( [row + ["spam"] for row in sp_body_spam[n] ] )
		writer.writerow( [ ] ) # Empty line
	f.close()

#========|| Perform over k-groups ||========#

# Output accuracy
#	Given data split into k-groups for both legit and spam
#	Train data on k-1 groups
#	Test classifier on remaining group
#	Print average accuracy of classifier after 10 iterations
def output_accuracy(sp_legit, sp_spam):

	sum_accuracy = 0.0

	for test_num in range( 10 ):

		mean_legit, mean_spam = [], []
		sd_legit, sd_spam = [], []

		# Combine 9/10 of the data to create training data
		train_legit = []
		train_spam = []

		for training_num in range( FOLD ):
			if training_num != test_num:
				train_legit.extend( sp_legit[training_num] )
				train_spam.extend(  sp_spam[training_num]  )

		# Train data
		mean_legit, mean_spam, sd_legit, sd_spam = train( train_legit, train_spam )

		# Checking accuracy of classifier on test data
		accuracy = classify( sp_legit[test_num], sp_spam[test_num], mean_legit, mean_spam, sd_legit, sd_spam )

		print( "\tTest on fold #{}: {}%".format( test_num, round(accuracy * 100, 2) ) )

		sum_accuracy += accuracy

	print( "\n\tAverage of accuracies: {}%\n".format( round((sum_accuracy / FOLD) * 100, 2) ) )
	return sum_accuracy / FOLD * 100

#===| Classify Subject corpus using Naive Bayes |===#

HIGH_DENSITY = 8.0    # P.D. with Laplace correction X U {0.065}
LOW_DENSITY	 = 1e-50 # P.D. for when exponential is 0
TINY_DENSITY = 1e-250 # P.D. for extremely unlikely i.e. stdev = 0

print("Accuracy of Classifier on Subject Corpus\n")
prob = output_accuracy(sp_subj_legit, sp_subj_spam)

#===| Classify Body corpus using Naive Bayes |===#

HIGH_DENSITY = 116.0  # P.D. with Laplace correction X U {0.065}
LOW_DENSITY	 = 1e-100 # P.D. for when exponential is 0
TINY_DENSITY = 1e-250 # P.D. for extremely unlikely i.e. stdev = 0

print("Accuracy of Classifier on Body Corpus\n")
output_accuracy(sp_body_legit, sp_body_spam)