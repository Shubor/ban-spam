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
c_subj_legit = []
c_subj_spam  = []
c_body_legit = []
c_body_spam  = []

# Stores document frequency score for each word
# 	The number of documents in which the term occurs
subj_corpus = Counter()
body_corpus = Counter()

# Clean:
# 	Check if word is not irrelevant
# 	Returns 1 if irrevelant, otherwise 0
def clean(word):
	# Remove if word is punctuation or if word is in stop words
	if word in punctuation or word.isdigit() or word in stop_words:
		return 1
	# Remove strings that contain digits e.g. 'qwoie192kjwe' or '12312-123'
	elif re.compile('\d').search(word):
		return 1
	else:
		return 0

# TD-IDF:
#	Calculates td-idf of term tk in document dj
#	Returns positive float if tk in dj; 0 otherwise
def TFIDF(term, n_term_in_doc, logTk):
	return n_term_in_doc[term]*(logT - logTk[term])

# Cosine Norm:
#	Calculates cosine norm for term tk in document dj
#	Return 0 if tk not in dj; value in [0,1] otherwise
def CosineNorm(term, n_term_in_doc, corpus, logTk):
	denominator = 0
	for word in n_term_in_doc:
		denominator += pow(TFIDF(word, n_term_in_doc, logTk), 2)
	if denominator == 0:
		return 0
	else:
		return TFIDF(term, n_term_in_doc, logTk)/math.sqrt(denominator)

# Write to file:
# 	Writes cosine norms to csv file, adding class of file to end of each row
def write_file(writer, cosine_norms, file_class):
	if file_class == "spam":
		for item in cosine_norms:
			item.append("spam")
			writer.writerow(item)
	else:
		for item in cosine_norms:
			item.append("non-spam")
			writer.writerow(item)

# Read in stop words from file
stop_words = []
with open('english.stop', 'rU') as stop_file:
	stop_words = [line.rstrip('\n') for line in stop_file]

# Append words to either subject or body list
for file in os.listdir(path):
	num_files += 1
	f = gzip.open(os.path.join(path, file), 'rb')

	# Counter data structure to store DF of each word in current file
	curr_body_corpus = Counter()
	curr_subj_corpus = Counter()

	try:
		for line in f:
			# Replace punctuation with space
			l = re.sub('[%s]' % re.escape(punctuation), ' ', line)

			# Subject corpus
			if line.startswith('Subject:'):
				for word in l.split():
					if not clean(word):
						if word != 'Subject':
							curr_subj_corpus[word] += 1

			# Body corpus
			else:
				for word in l.split():
					if not clean(word):
						curr_body_corpus[word] += 1

		# Update number of documents with term word i.e. #Tr(t_k)
		for word in curr_body_corpus:
			body_corpus[word] += 1
		for word in curr_subj_corpus:
			subj_corpus[word] += 1

		# Counter of ALL the terms of the document
		if "spmsg" in file:
			c_body_spam.append(curr_body_corpus)
			c_subj_spam.append(curr_subj_corpus)
		else:
			c_body_legit.append(curr_body_corpus)
			c_subj_legit.append(curr_subj_corpus)


	finally:
		f.close()

#===[Equation Variables]===#
# log(|T|)
logT = math.log(num_files)

# log(#T(tk)) - body
logTk_body = {}
for tk in body_corpus:	# ALL terms
	logTk_body[tk] = math.log(body_corpus[tk])

# log(#T(tk)) - subject
logTk_subj = {}
for tk in subj_corpus:	# ALL terms
	logTk_subj[tk] = math.log(subj_corpus[tk])
#===[Equation Functions]===#

#======[2D-array of cosine norm]======#
w_body_legit= []
w_body_spam = []

for document in c_body_legit:
	row = []
	for term in body_corpus.most_common(N):
		row.append( CosineNorm( term[0], document, body_corpus, logTk_body ))
	w_body_legit.append(row)

for document in c_body_spam:
	row = []
	for term in body_corpus.most_common(N):
		row.append( CosineNorm( term[0], document, body_corpus, logTk_body ))
	w_body_spam.append(row)

w_subj_legit= []
w_subj_spam = []

for document in c_subj_legit:
	row = []
	for term in subj_corpus.most_common(N):
		row.append( CosineNorm( term[0], document, subj_corpus, logTk_subj ))
	w_subj_legit.append(row)

for document in c_subj_spam:
	row = []
	for term in subj_corpus.most_common(N):
		row.append( CosineNorm( term[0], document, subj_corpus, logTk_subj ))
	w_subj_spam.append(row)

#====[Output 2D-array as CSV]====#
header = [ "f" + str(x) for x in range(1, N+1) ]
header.append("class")

with open("body.csv", "wb") as f:

	writer = csv.writer(f)
	writer.writerow(header)

	write_file(writer, w_body_legit, "nonspam")
	write_file(writer, w_body_spam, "spam")

	f.close()

with open("subject.csv", "wb") as f:

	writer = csv.writer(f)
	writer.writerow(header)

	write_file(writer, w_subj_legit, "nonspam")
	write_file(writer, w_subj_spam, "spam")

	f.close()





