import csv
import math


N = 200
FOLD = 10

NONSPAM = "nonspam"
SPAM 	= "spam"


#======|| Read in csv files ||======#

w_body_legit, w_body_spam = [], []
w_subj_legit, w_subj_spam = [], []

# Read CSV:
#	Read in given csv file
#	Saves data from file into either legit or spam lists
#		depending on class of file given by last column of csv
def read_csv( file_name, legit, spam ):

	with open( file_name, "r" ) as f:
		for row in csv.reader(f):
			# Append 
			if row[-1] == NONSPAM:
				legit.append( row[:-1] )
			elif row[-1] == SPAM:
				spam.append(  row[:-1] )

read_csv( "body.csv",    w_body_legit, w_body_spam )
read_csv( "subject.csv", w_subj_legit, w_subj_spam )

#=====================================================#
# Calculate Naive Bayes, assuming normal distribution #                      #
#=====================================================#

#======|| Calculate Mean ||======#

# Mean:
def mean( column ):

	return math.fsum( [float(row) for row in column] ) / len( column )

#======|| Calculate Standard Deviation ||======#

# Standard deviation of a sample:
def std_dev( column ):

	u = mean(column)
	return math.sqrt( math.fsum([((float(x) - u) ** 2) for x in column]) / ( len(column) - 1 ) )

#======|| Probability of interval ||======#

c = 30
MIN = 0.0
MAX = 0.05
DEF_STDEV = 0.7 * (MAX - MIN) / k

# Probability of P(x < a) assuming Normal Distribution
def phi( a, u, s ):

	if s == 0:
		s = DEF_STDEV

	return 0.5 * (1 + math.erf((x - u)/math.sqrt(2 * s ** 2)))

# Probability of the interval (a,b] given a Normal Distribution.
def P( a, b, u, s ):

	return phi(b,u,s) - phi(a,u,s)

# Calculate the Euclidean Norm of the vectors x and y.
def norm( x, y ):

	d = 0.0

	for i in len(x):
		d += ( x[i] - y[i] )**2

	return math.sqrt(d)

# Calculate the dot product of two lists
def dot_product( x, y ):

	product = 0.0

	for i in range(len(x)):
		product += x[i] * y[i]

	return product


#=======|| Initialise degree of membership ||=======#

# The degree of membership for data point i to cluster j
# is initialised with a random value 0 <= a_ij <= 1
# such that \Sigma^{C}_{j} \lambda_ij = 1
import random

# Generate an NxC matrix of random values where the column sums to 1.
def generate_membership( stored, N, C ):
	
	for n in range(N):	

		stored.append([])

		rndm = sorted([random.random() for foo in range(C-1)])

		stored[n].append( rndm[0] )
		[stored[n].append( rndm[k] - rndm[k-1] ) for k in range( 1, C-1 )]
		stored[n].append( 1 - rndm[-1] )

		random.shuffle(stored[n])

#=======|| Degree of membership ||=======#
def member_of_degree( x, j, cluster_centres, m ):

	degree = 0.0
	numerator 	= norm( x, cluster_centres[j] )

	if numerator == 0:
		return -1

	power = 2/(m-1)

	for k in range(len(cluster_centres)):

		denominator = norm( x, cluster_centres[k] )
		if denominator == 0:
			return 0

		degree += math.pow( numerator / denominator, power )

# For a given data point x_i, the degree of its membership to cluster j is calculated as follows:
def membership( data_points, cluster_centres, memberships, m ):

	for i in len(data_points):
		for j in len(cluster_centres):

			memberships[i][j] = member_of_degree( data_points[i], j, cluster_centres, m )



# In each iteration of the FCM algorithm, the following objective function is minimised
def objective_func( data_points, degree_of_membership, cluster_centres ):
	J = 0

	for i in len(data_points):
		for j in len(cluster_centres):

			J += degree_of_membership[i][j] *  norm( data_points[i], cluster_centres[j] )

#========|| Calculate centre vector ||=========#
def centre( data_points, degree_of_membership, m ):



#======|| Probability Density Function ||======#

# Probability density function
#	Calculates probability of x assuming normal distribution
#	x = value; u = mean; s = standard deviation
# 	returns float
test_value = 1e-120
def prob_density( x, u, s ):

	if s == 0.0 and x == u:
		return 1.0

	elif s == 0.0 and x != u:
		return 1e-38

	coefficient = 1.0 / ( s * math.sqrt( 2.0 * math.pi ) )
	exponent = - ((x - u) ** 2.0) / (2.0 * (s ** 2.0))

	density = coefficient * math.exp(exponent)
	if density == 0:
		return 1e-50

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
# 	Get P(spam|document) = P(f1|spam)*.....*P(f200|spam)
# 	Similarly, calculate P(nonspam|document)
#	Compare P(spam|document) and P(nonspam|document)
#	If there is a tie, choose non-spam
#		Should implement a threshold for this to avoid false positives
def classify( test_legit, test_spam, mean_legit, mean_spam, sd_legit, sd_spam ):

	# Number of examples = 60, prob_spam = number of examples which are spam / examples = 20/60
	TOTAL_DOCS = float( len(test_spam) + len(test_legit) ) # Number of examples = 60
	P_spam  = len( test_spam )  / TOTAL_DOCS # P(SPAM) is |SPAM|/|EXAMPLES| = 0.33334
	P_legit = len( test_legit ) / TOTAL_DOCS # P(NONSPAM) is 1-P(SPAM) = 0.6666667
	SPAM = "spam"
	LEGIT = "nonspam"
	num_correct = 0

	# Lp = 0.05 # Laplace correction used by weka
	C = 1.0 # Threshold for spam: P(spam|X=x) > C P(C=legit|X=x)

	# Test on the known legitimate documents
	for document in test_legit:

		P_legit_X, P_spam_X = 0.0, 0.0

		# Calculate P(class|X)=P(X1|class) ... P(X200|class) P(class)
		for x in range(len(document)):
			P_X_spam = prob_density( float(document[x]), mean_spam[x] , sd_spam[x]  )
			P_X_legit = prob_density( float(document[x]), mean_legit[x], sd_legit[x] )

			if P_X_spam != 0.0:
				P_spam_X  += math.log(P_X_spam)
			else:
				P_spam_X += math.log(test_value)
			if P_X_legit != 0.0:
				P_legit_X += math.log(P_X_legit)
			else:
				P_legit_X += math.log(test_value)

		# P(class|X) = P(X|class) P(class)
		P_legit_X += math.log(P_legit)
		P_spam_X  += math.log(P_spam)

		# Classify
		if C * P_legit_X >= P_spam_X:
			num_correct += 1
		else:
			print("1: ",P_legit_X,P_spam_X)

	# Test on the known spam documents
	for document in test_spam:
		P_legit_X, P_spam_X = 0.0, 0.0

		# Calculate P(class|X)=P(X1|class) ... P(X200|class) P(class)
		for x in range(len(document)):
			P_X_spam = prob_density( float(document[x]), mean_spam[x] , sd_spam[x]  )
			P_X_legit = prob_density( float(document[x]), mean_legit[x], sd_legit[x] )

			if P_X_spam != 0.0:
				P_spam_X  += math.log(P_X_spam)
			else:
				P_spam_X += math.log(test_value)
			if P_X_legit != 0.0:
				P_legit_X += math.log(P_X_legit)
			else:
				P_legit_X += math.log(test_value)

		# P(class|X) = P(X|class) P(class)
		P_legit_X += math.log(P_legit)
		P_spam_X  += math.log(P_spam)

		# Classify
		if C * P_legit_X < P_spam_X:
			num_correct += 1
		else:
			print("2: ",P_legit_X,P_spam_X)

	# Accuracy
	return num_correct / TOTAL_DOCS

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



#========|| Perform over K-groups ||========#
max_accuracy = 0.0
best_val = 0.000000000

# print("Accuracy of Naive on folds")

while max_accuracy < 90.0:
	# Iterate for each for fold
	
	sum_accuracy = 0.0
	for test_num in range( 10 ):

		mean_legit, mean_spam = [], []
		sd_legit, sd_spam = [], []

		# Create training data from other 9/10 of the examples.
		train_legit = []
		train_spam = []

		for training_num in range( FOLD ):
			if training_num != test_num:
				train_legit.extend( sp_body_legit[training_num] )
				train_spam.extend(  sp_body_spam[training_num]  )

		# Train data
		# mean_legit[9] is the mean of 1:360 => test_legit is 361:400
		mean_legit, mean_spam, sd_legit, sd_spam = train( train_legit, train_spam )

		# sp_body_legit[9] is 361:400
		# works for mean_legit and sd_legit

		# Checking accuracy on test data 361:400
		accuracy = classify( sp_body_legit[test_num], sp_body_spam[test_num], mean_legit, mean_spam, sd_legit, sd_spam )

		print( "Test on fold #{}: {}%".format( test_num, round(accuracy * 100, 2) ) )

		sum_accuracy += accuracy

	print("\nAverage of accuracies: {}%".format( round((sum_accuracy / FOLD) * 100, 2) ))
	
	
	if sum_accuracy*10 >= max_accuracy:
		best_val = test_value
		max_accuracy = sum_accuracy*10
		print(best_val,max_accuracy)
	test_value *= 1000