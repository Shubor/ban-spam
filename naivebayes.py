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
def standard_dev( column ):

	u = mean(column)
	return math.sqrt( math.fsum( [ ((float(x) - u) ** 2) for x in column] ) / ( len(column) - 1 ) )

#======|| Probability Density Function ||======#

# Probability density function
#	Calculates probability of x assuming normal distribution
#	x = value; u = mean; s = standard deviation
# 	returns float
def prob_density( x, u, s ):

	if s == 0:
		return 0.0
	coefficient = math.e / ( s * math.sqrt( 2 * math.pi ) )
	exponent = -( (x - u) ** 2) / (2 * (s ** 2))
	return coefficient ** exponent

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
def split( data, fold ):

	n = len( data ) / float( fold )
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
		mean_legit.append( mean( [row[n] for row in train_legit] ) )
		sd_legit.append( standard_dev( [row[n] for row in train_legit] ) )

	for n in range(len(train_spam[0])):
		mean_spam.append( mean( [row[n] for row in train_spam] ) )
		sd_spam.append( standard_dev( [row[n] for row in train_spam] ) )

	return mean_legit, mean_spam, sd_legit, sd_spam

# Classify:
# 	Get P(spam|document) = P(f1|spam)*.....*P(f200|spam)
# 	Similarly, calculate P(nonspam|document)
#	Compare P(spam|document) and P(nonspam|document)
#	If there is a tie, choose non-spam
#		Should implement a threshold for this to avoid false positives
def classify( test_legit, test_spam, mean_legit, mean_spam, sd_legit, sd_spam ):

	TOTAL_DOCS = float( len(test_spam) + len(test_legit) )
	PROB_SPAM = len( test_spam ) / TOTAL_DOCS
	PROB_LEGIT = len( test_legit ) / TOTAL_DOCS
	SPAM = "spam"
	LEGIT = "nonspam"

	num_correct = 0

	for document in test_legit:

		spam_vals, legit_vals = 1.0, 1.0
		col_num = 0

		for x in document:
			# Calculate P(f1|spam).....P(f200|spam) for document
			spam_vals  *= prob_density( float(x), mean_spam[col_num], sd_spam[col_num] )   + 1
			# Calculate P(f1|nonspam).....P(f200|nonspam) for document
			legit_vals *= prob_density( float(x), mean_legit[col_num], sd_legit[col_num] ) + 1

			col_num += 1

		spam_vals  *= PROB_SPAM
		legit_vals *= PROB_LEGIT

		if legit_vals >= spam_vals:
			num_correct += 1

	for document in test_spam:

		spam_vals, legit_vals = 1.0, 1.0
		col_num = 0

		for x in document:
			spam_vals  *= prob_density( float(x), mean_spam[col_num], sd_spam[col_num] )   + 1
			legit_vals *= prob_density( float(x), mean_legit[col_num], sd_legit[col_num] ) + 1

			col_num += 1

		spam_vals  *= PROB_SPAM
		legit_vals *= PROB_LEGIT

		if spam_vals > legit_vals:
			num_correct += 1

	# Accuracy
	return num_correct / TOTAL_DOCS

#=========|| 10-FOLD stratified cross validation ||=========#

split_body_legit = split( w_body_legit, FOLD )
split_body_spam  = split( w_body_spam,  FOLD )
split_subj_legit = split( w_subj_legit, FOLD )
split_subj_spam  = split( w_subj_spam,  FOLD )

sum_accuracy = 0.0

test_num = 0

while test_num != FOLD:
	mean_legit, mean_spam = [], []
	sd_legit, sd_spam = [], []

	# Add all the training data together
	train_legit = []
	train_spam = []
	for training_num in range( 0, FOLD ):
		if training_num != test_num:
			train_legit.extend( split_body_legit[training_num] )
			train_spam.extend(  split_body_spam[training_num]  )

	# Train data
	mean_legit, mean_spam, sd_legit, sd_spam = train( train_legit, train_spam )

	# Test data
	accuracy = classify( split_body_legit[test_num], split_body_spam[test_num], mean_legit, mean_spam, sd_legit, sd_spam )
	sum_accuracy += accuracy

	test_num += 1

print sum_accuracy/10.0

