import csv

#======|| Read in csv files ||======#

w_body_legit, w_body_spam = [], []
w_subj_legit, w_subj_spam = [], []

# Read CSV:
#	Read in given csv file
#	Saves data from file into either legit or spam lists
#		depending on class of file given by last column of csv
def read_csv(file_name, legit, spam):
	with open(file_name, "rb") as f:
		row_num = 0
		for row in csv.reader(f):
			# Ignore header
			if row_num != 0:
				# Check class of file (stored in last row of csv)
				if row[-1] == "nonspam":
					# Append row, removing class of doc
					legit.append(row[:-1])
				elif row[-1] == "spam":
					spam.append(row[:-1])
			row_num += 1

read_csv("body.csv", w_body_legit, w_body_spam)
read_csv("subject.csv", w_subj_legit, w_subj_spam)


#############################################################################
# Calculate Naive Bayes, assuming normal distribution                       #
#############################################################################

import math

#======|| Calculate Mean ||======#

mean_body_legit, mean_body_spam = [], []
mean_subj_legit, mean_subj_spam = [], []

# mean: calculates the average value of a set
# returns: float
def mean(column):
	return math.fsum([row for row in column]) / len(column)

# calculate the mean of column fn
for n in range(N):
	mean_body_legit.append( mean([row[n] for row in w_body_legit] ))
	mean_body_spam.append(  mean([row[n] for row in w_body_spam]  ))

	mean_subj_legit.append( mean([row[n] for row in w_subj_legit] ))
	mean_subj_spam.append(  mean([row[n] for row in w_subj_spam]  ))

sd_body_legit, sd_body_spam = [], []
sd_subj_legit, sd_subj_spam = [], []

#======|| Calculate Standard Deviation ||======#

# standard_dev: calculates the standard deviation of a set
# returns: float
def standard_dev(column):
	u = mean(column)
	return math.sqrt( math.fsum([pow(x-u, 2) for x in column]) / (len(column)-1) )

# calculate the standard deviation of column fn
for n in range(N):
	stdev_body_legit.append( standard_dev( [row[n] for row in w_body_legit] ))
	stdev_body_spam.append(  standard_dev( [row[n] for row in w_body_spam ] ))
	stdev_subj_legit.append( standard_dev( [row[n] for row in w_subj_legit] ))
	stdev_subj_spam.append(  standard_dev( [row[n] for row in w_subj_spam ] ))

#======|| Probability Denisty Function ||======#

# Probability density function: calculates probability of x assuming normal distribution
# Return: float
def prob_density( x, u, s ):
	coefficient = pow(s*math.sqrt(2*math.pi),-1)	
	exponent = (-1) * pow(x-u, 2) / (2 * pow(s,2))
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

