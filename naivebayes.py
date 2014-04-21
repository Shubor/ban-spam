import csv

#############################################################################
# Read in csv files                                                         #
#############################################################################

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

mean_body_legit, mean_body_spam = [], []
mean_subj_legit, mean_subj_spam = [], []

# Mean:
#	Mean is the average of given values
#	E[X] := sum_values / count_values
def mean(column):
	return math.fsum([row for row in column]) / len(column)

for n in range(N):
	# Calculate mean of column fn which are legit
	mean_body_legit.append( mean([row[n] for row in w_body_legit] ))
	mean_body_spam.append(  mean([row[n] for row in w_body_spam]  ))

	mean_subj_legit.append( mean([row[n] for row in w_subj_legit] ))
	mean_subj_spam.append(  mean([row[n] for row in w_subj_spam]  ))

sd_body_legit, sd_body_spam = [], []
sd_subj_legit, sd_subj_spam = [], []

# Standard deviation:
#	SD of X => Sqrt( E[X^2] - (E[X])^2 )
def standard_dev(column):
	u = mean(column)
	return math.sqrt( math.fsum([pow(x-u, 2) for x in column]) / (len(column)-1) )

for n in range(N):
	# Calculate mean of column fn which are legit
	stdev_body_legit.append( standard_dev( [row[n] for row in w_body_legit] ))
	stdev_body_spam.append(  standard_dev( [row[n] for row in w_body_spam ] ))
	stdev_subj_legit.append( standard_dev( [row[n] for row in w_subj_legit] ))
	stdev_subj_spam.append(  standard_dev( [row[n] for row in w_subj_spam ] ))

# Probability density function:
#	Given by mean and standard deviation for a normal distribution
def prob_density(mean, std_dev):
	x = math.e / (std_dev * math.sqrt(2*math.pi))
	power = -( math.pow((mean-std_dev), 2) / (2 * math.pow(std_dev, 2)) )
	return = math.pow(x, power)

