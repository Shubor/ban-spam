import csv
import math

N = 200
ten_fold = 10

#======|| Read in csv files ||======#

w_body, w_subj = [], []

w_body_legit, w_body_spam = [], []
w_subj_legit, w_subj_spam = [], []

# Read CSV:
#	Read in given csv file
#	Saves data from file into either legit or spam lists
#		depending on class of file given by last column of csv
def read_csv(file_name, legit, spam, data):
	with open(file_name, "rb") as f:
		row_num = 0
		for row in csv.reader(f):
			# Ignore header
			if row_num != 0:
				data.append(row)
				# Check class of file (stored in last row of csv)
				if row[-1] == "nonspam":
					# Append row, removing class of doc
					legit.append(row[:-1])
				elif row[-1] == "spam":
					spam.append(row[:-1])
			row_num += 1

read_csv("body.csv", w_body_legit, w_body_spam, w_body)
read_csv("subject.csv", w_subj_legit, w_subj_spam, w_subj)

print w_body_legit[0]
print w_body[0]

#############################################################################
# Calculate Naive Bayes, assuming normal distribution                       #
#############################################################################

#======|| Calculate Mean ||======#

mean_body_legit, mean_body_spam = [], []
mean_subj_legit, mean_subj_spam = [], []

# Mean:
#	Calculates the average value of a set
# 	returns float
def mean(column):
	return math.fsum([float(row) for row in column]) / len(column)

# Calculate the mean of column fn
for n in range(N):
	mean_body_legit.append( mean([row[n] for row in w_body_legit] ))
	mean_body_spam.append(  mean([row[n] for row in w_body_spam]  ))
	mean_subj_legit.append( mean([row[n] for row in w_subj_legit] ))
	mean_subj_spam.append(  mean([row[n] for row in w_subj_spam]  ))

#======|| Calculate Standard Deviation ||======#

# s = sqrt( 1/(|X|-1) * sum((x-u)^2) for all points,x, in data, X )

sd_body_legit, sd_body_spam = [], []
sd_subj_legit, sd_subj_spam = [], []

# standard_dev
#	Calculates the standard deviation of a set
# 	returns: float
def standard_dev(column):
	u = mean(column)
	return math.sqrt( math.fsum([pow(float(x)-u, 2) for x in column]) / (len(column)-1) )

# Calculate the standard deviation of column fn
for n in range(N):
	sd_body_legit.append( standard_dev( [row[n] for row in w_body_legit] ))
	sd_body_spam.append(  standard_dev( [row[n] for row in w_body_spam ] ))
	sd_subj_legit.append( standard_dev( [row[n] for row in w_subj_legit] ))
	sd_subj_spam.append(  standard_dev( [row[n] for row in w_subj_spam ] ))

#======|| Probability Density Function ||======#

# Probability density function
#	Calculates probability of x assuming normal distribution
#	x = value; u = mean; s = standard deviation
# 	returns float
def prob_density(x, u, s):
	if s == 0:
		return 0.0
	coefficient = pow(s * math.sqrt(2 * math.pi), -1)
	exponent = (-1) * pow(x-u, 2) / (2 * pow(s, 2))
	return coefficient * math.expm1( exponent )

#=========|| Naive Bayes ||=========#
# f(x=x1 | spam ) > f(x=x1 | nonspam)?

# Given: a vector of cosine normed tdidf values of top200 document set terms
#
# Find:	document frequency of top200 terms
#	vector of cosine normed tdidf values of top200 document set terms
# Method: apply naive bayes using above vector for spam and nonspam
#	  choose "nonspam" for ties
# Return: "spam" or "nonspam"

# Split:
#	Split data into 10 even chunks
def split(data):
	n = len(data)/float(ten_fold)
	out = []
	last = 0.0
	while last < len(data):
		out.append(data[int(last):int(last+n)])
		last += n
	return out

split_body = split(w_body)
split_subj = split(w_subj)

# Train:
#	Calculate mean and standard deviation of data based on given training data
def train(training_data):
	legit, spam = [], []
	mean_legit, mean_spam = [], []
	sd_legit, sd_spam = [], []
	# Separate documents into legit or spam
	for document in training_data:
		if document[-1] == "nonspam":
			legit.append(document[:-1])
		else:
			spam.append(document[:-1])
	# Calculate mean and standard deviation of column fn
	for n in range(len(legit[0])):
		mean_legit.append(mean([row[n] for row in legit]))
		sd_legit.append(standard_dev([row[n] for row in legit]))
	for n in range(len(spam[0])):
		mean_spam.append(mean([row[n] for row in spam]))
		sd_spam.append(standard_dev([row[n] for row in spam]))
	return mean_legit, mean_spam, sd_legit, sd_spam

# Classify:
# 	Get P(spam|document) = P(f1|spam)*.....*P(f200|spam)
# 	Similarly, calculate P(nonspam|document)
#	Compare P(spam|document) and P(nonspam|document)
#	If there is a tie, choose non-spam
#		Should implement a threshold for this to avoid false positives
def classify(test_data, mean_legit, mean_spam, sd_legit, sd_spam):
	num_correct = 0
	for document in test_data:
		prob_spam, prob_nonspam = 1.0, 1.0
		col_num = 0
		for i in document[:-1]: # Exclude class of document
			# Calculate P(f1|spam).....P(f200|spam) for document
			prob_spam *= prob_density(float(i), mean_spam[col_num], sd_spam[col_num])+1
			# Calculate P(f1|nonspam).....P(f200|nonspam) for document
			prob_nonspam *= prob_density(float(i), mean_legit[col_num], sd_spam[col_num])+1
			col_num += 1
		prediction = ""
		if prob_nonspam >= prob_spam:
			prediction = "nonspam"
		else:
			prediction = "spam"
		if document[-1] == prediction:
			num_correct += 1
	accuracy = num_correct/float(len(test_data))
	return accuracy

#=========|| 10-fold stratified cross validation ||=========#
# Cross validation
#	Split data into 10 subsets
#	Classifier built ten_fold times
#		Each time the testing is on 1 segment, and training on remaining ten_fold-1
#	Average accuracies of each run to calc overall accuracy
# Repeat 10 times

sum_accuracy = 0.0

test_num = 0

while test_num != ten_fold:
	mean_legit, mean_spam = [], []
	sd_legit, sd_spam = [], []

	# Add all the training data together
	training_data = []
	for training_num in range(0, ten_fold):
		if training_num != test_num:
			training_data.extend(split_body[training_num])

	# Train data
	mean_legit, mean_spam, sd_legit, sd_spam = train(training_data)

	# Test data
	accuracy = classify(split_body[test_num], mean_legit, mean_spam, sd_legit, sd_spam)
	sum_accuracy += accuracy

	test_num += 1

print sum_accuracy/10.0

