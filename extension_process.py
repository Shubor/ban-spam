import os
import re
import csv
import gzip
import math
from string import digits, punctuation
from collections import Counter

path = 'lingspam-mini600'

# Select only top N words
N = 200

# Total number of files, |T|, in the document set
num_files = 0

# List of subject and body counters for each file
# 	Stores top N terms in a counter, value=freq in file

# A list of counters. Each counter corresponds to a file, with the count of each word
# of that file.
c_body_legit, c_body_spam = [], []
c_subj_legit, c_subj_spam = [], []

# Stores document frequency score for each word
# 	The number of documents in which the term occurs
body_corpus = Counter()
subj_corpus = Counter()

# Clean:
# 	Check if word is not irrelevant
# 	Returns 1 if irrevelant, otherwise 0
def clean( word ):

	# Remove if word is punctuation or if word is in stop words
	if word in stop_words:
		return 1
	# Remove strings that contain digits e.g. 'qwoie192kjwe' or '12312-123'
	elif re.compile('\d').search(word):
		return 1
	else:
		return 0

# Read in stop words from file
stop_words = []
with open('english.stop', 'rU') as stop_file:
	stop_words = [ line.rstrip('\n') for line in stop_file ]

#===========================================================================#
# Read in emails and compute document frequency scores                      #
#===========================================================================#

# A list of counters. Each counter corresponds to a file, with the count of each word
# of that file.
tf_body_legit = Counter()
tf_body_spam  = Counter()

tf_subj_legit = Counter()
tf_subj_spam  = Counter()

use = 0
useline = ''
# Append words to either subject or body list
for file in os.listdir(path):

	num_files += 1
	f = gzip.open(os.path.join(path, file), "rb")

	# Counter data structure to store DF of each word in current file
	curr_body_corpus = Counter()
	curr_subj_corpus = Counter()

	# Counters count frequency of all words in current file
	terms_body_spam = Counter()
	terms_subj_spam = Counter()

	terms_body_legit = Counter()
	terms_subj_legit = Counter()


	try:
		for line in f:

			# Replace punctuation with space
			line = line.decode("utf-8")
			l = re.sub('[%s]' % re.escape(punctuation), " ", line)
			# Subject corpus
			if line.startswith("Subject:"):

				for word in l.split():
					if not clean(word):
						if word != 'Subject' and "spmsg" in file:						
							curr_subj_corpus[word] += 1
							terms_subj_spam[word]  += 1
						
						elif word != 'Subject':						
							curr_subj_corpus[word] += 1
							terms_subj_legit[word] += 1
			
			# Body corpus
			else:
				
				for word in l.split():
					if not clean(word) and "spmsg" in file:

						curr_body_corpus[word] += 1
						terms_body_spam[word]  += 1

					elif not clean(word):

						curr_body_corpus[word] += 1
						terms_body_legit[word] += 1						


		# Update number of documents with term word i.e. #Tr(t_k)
		for word in curr_body_corpus:
			body_corpus[word] += 1
		for word in curr_subj_corpus:
			subj_corpus[word] += 1

		# Add binary occurence of terms to the corpus's term frequency counter 
		
		for term in terms_body_legit:
			tf_body_legit[term] += 1

		for term in terms_body_spam:
			tf_body_spam[term]  += 1

		for term in terms_subj_legit:
			tf_subj_legit[term] += 1

		for term in terms_subj_spam:			
			tf_subj_spam[term]	+= 1

		# if "spmsg" in file:
			
		# 	if line.startswith("Subject:"):
		
		# 		for word in curr_subj_corpus:
		# 			tf_subj_spam[word] += 1
		
		# 	else:
		
		# 		for word in curr_body_corpus:
		# 			tf_body_spam[word] += 1

		# else:

		# 	if line.startswith("Subject:"):
		
		# 		for word in curr_subj_corpus:
		# 			tf_subj_legit[word] += 1
		
		# 	else:
		
		# 		for word in curr_body_corpus:
		# 			tf_body_spam[word] += 1

		# Counter of ALL the terms of the document

		if "spmsg" in file:
			c_body_spam.append( curr_body_corpus )
			c_subj_spam.append( curr_subj_corpus )
		else:
			c_body_legit.append( curr_body_corpus )
			c_subj_legit.append( curr_subj_corpus )

	finally:
		f.close()

#############################################################################
# Feature weighting with tf-idf, then cosine normalisation of top 200 words #
#############################################################################

# |T|
T = num_files

# log(|T|)
logT = math.log2(T)

# log(#T(tk)) of body and subject
logTk_body = {}
logTk_subj = {}

for tk in body_corpus:	# ALL terms
	logTk_body[tk] = math.log2( body_corpus[tk] )
for tk in subj_corpus:	# ALL terms
	logTk_subj[tk] = math.log2( subj_corpus[tk] )

# TD-IDF:
#	Calculates td-idf of term tk in document dj
#	Returns positive float if tk in dj; 0 otherwise
def TFIDF( term, TF, logTk ):

	return TF[term] * ( logT - logTk[term] )


#============|| Feature Selection ||==============#

def IG_Error( x, y ):

	X_on_top = x / ( x + y )
	Y_on_top = y / ( x + y )

	return - (X_on_top * math.log2( X_on_top ) + Y_on_top * math.log2( Y_on_top ))

def Information_Gain( n_category, term_freq, n):
	
	IG_values = Counter()

	for category in range( len(term_freq) ):

		for term in term_freq[category]:

			# Number of documents
			N = sum(n_category)

			# Number of documents of category with term 
			A = term_freq[category][term]

			# Number of documents of not category with term
			B = term_freq[category][term]

			# Number of category without term
			C = n_category[category] - A

			# Number of documents of not category without term
			# D = n_category[-category] - B
			D = N - A - B - C

			POS = A + C
			NEG = B + D

			PW 	= POS / N
			PNW = 1 - PW

def CPD( n_category, term_freq, d, n ):

	CPD_values = Counter()

	for category in range( len(term_freq) ):

		for term in term_freq[category]:

			# Number of documents of category with term 
			A = term_freq[category][term]

			# Number of documents of not category with term
			B = term_freq[1-category][term]

			numerator 	= float( A - B )
			denominator = float( A + B )

			# print("term: {} with {} on {} is {}".format(term,numerator,denominator,numerator/denominator))


			if A-B > d:
				CPD_values[term] = max( CPD_values[word], numerator/denominator )

	return CPD_values.most_common(n)



# term_freq is a list of two counters; each gives the term frequency
# of words that appeared in the given corpus

def chi( n_category, term_freq, n ):

	chi_values = Counter()

	for category in range( len(term_freq) ):

		for term in term_freq[category]:

			# Number of documents of category with term 
			A = term_freq[category][term]

			# Number of documents of not category with term
			B = term_freq[1-category][term]

			# Number of category without term
			C = n_category[category] - A

			# Number of documents of not category without term
			# D = n_category[-category] - B
			D = sum(n_category) - A - B - C

			numerator = N * ( A*D - B*C ) ** 2
			denominator = (A + C) * (B + D) * (A + B) * (C + D)

			#print("term: {} with {} on {}".format(term,numerator,denominator))

			if denominator == 0:
				chi_values[term] = float('Infinity')
			else:
				chi_values[term] = max( chi_values[word], numerator/denominator )

	return chi_values.most_common(n)

def Mutual_Information( n_category, term_freq, n ):

	MI_values = Counter()

	for category in range( len(term_freq) ):

		for term in term_freq[category]:

			# Number of documents of category with term 
			A = term_freq[category][term]

			# Number of documents of not category with term
			B = term_freq[1-category][term]

			# Number of category without term
			C = n_category[category] - A

			N = sum(n_category)

			MI = math.log( (A * N) / ( (A + B) * (A + C) ) )

			print(term,A,N,B,C,MI)

			MI_values[term] = max( MI_values[word], MI )

	return MI_values.most_common(n)


#=======|| Cosine Normalisation ||========#

#	cosine_norm: calculates cosine norm for term tk in document dj
#	Return 0 if tk not in dj; value in [0,1] otherwise
def cosine_norm( term, TF, logTk ):

	denominator = 0

	for word in TF:
		denominator += ( TFIDF( word, TF, logTk ) ** 2 )
	
	if denominator == 0:
		return 0
	else:
		return TFIDF( term, TF, logTk ) / math.sqrt( denominator )

def cosine_normalisation( corpus_tfidf, corpus_features, logTk ):

	corpus_cosNorm = []

	for document in corpus_tfidf:

		# Row containing cosine normailsed values for the selected features
		doc_cosNorm = [] 

		# For each feature find the cosine normalised value
		for feature in corpus_features:

			doc_cosNorm.append( cosine_norm( feature[0], document, logTk ) )

		corpus_cosNorm.append(doc_cosNorm)

	return corpus_cosNorm


d1, d2 = 25, 3
body_features = Mutual_Information( [400,200], [tf_body_legit, tf_body_spam], 200 )
subj_features = Mutual_Information( [400,200], [tf_subj_legit, tf_subj_spam], 200 )

print(body_features)


#print(body_features)
#print(subj_features)
#====|Calculate cosine normalised values|====#

# List with column = feature, row = email, and each entry ('term',feature value)
cosnorm_body_legit = cosine_normalisation( c_body_legit, body_features, logTk_body )
cosnorm_body_spam  = cosine_normalisation( c_body_spam,  body_features, logTk_body )

cosnorm_subj_legit = cosine_normalisation( c_subj_legit, subj_features, logTk_subj )
cosnorm_subj_spam  = cosine_normalisation( c_subj_spam,  subj_features, logTk_subj )

#############################################################################
# Outputs normalised tf-idf weights into csv files                          #
#############################################################################

# Write CSV:
#	Write to given csv file
#	Saves data to file, adding class of file into last column of csv
def write_csv( file_name, legit, spam ):

	header = [ "f" + str(x) for x in range(1, N + 1) ]
	header.append("class")
	with open(file_name, "w") as f:
		writer = csv.writer(f)
		writer.writerow(header)
		writer.writerows( [row + ["nonspam"] for row in legit] )
		writer.writerows( [row + ["spam"] for row in spam] )
		f.close()

write_csv("body.csv"   , cosnorm_body_legit, cosnorm_body_spam)
write_csv("subject.csv", cosnorm_subj_legit, cosnorm_subj_spam)