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

subj_corpus = Counter()
body_corpus = Counter()

# Check if word is not irrelevant
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

# Calculates td-idf of term in Counter of doc, freq_in_doc
# 	Returns positive float, 0 if not in the doc
def TFIDF(term, freq_in_doc):
	return freq_in_doc[term] * ( math.log(len(body_corpus)) - math.log(body_corpus[term]) )

def CosineNorm(term, freq_in_doc):
	denominator = 0

	for word in freq_in_doc:
		denominator += pow( TFIDF(word,freq_in_doc),2 )

	return TFIDF(term,freq_in_doc)/math.sqrt(denominator)
		

# Read in stop words from file
stop_words = []
with open('english.stop', 'rU') as stop_file:
	stop_words = [line.rstrip('\n') for line in stop_file]

# List of subject and body counters for each file (stores top N terms in a counter, value=freq in file)
c_body = []
c_subj = []

# Append words to either subject or body list
for file in os.listdir(path):
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
						curr_subj_corpus[word] += 1

			# Body corpus
			else:			
				for word in l.split():
					if not clean(word):
						curr_body_corpus[word] += 1
		
		# Update number of documents with term word i.e. #Tr(t_k)
		for word in curr_body_corpus:
			body_corpus[word]+=1
		for word in curr_subj_corpus:
			subj_corpus[word]+=1
		
		# Counter of ALL ther terms of the document
		c_body.append(curr_body_corpus)
		c_subj.append(curr_subj_corpus)

	finally:
		f.close()

for terms in c_body:	# terms is Counter (curr_body_corpus) of each file
	for term in terms.most_common(N):
		if terms.most_common(N)[0][1] > 1:
			print term[0],CosineNorm(term[0],terms),TFIDF(term[0],terms)
	print "<---------------------------->"


# Normalise using cosine normalisation
# Save data in csv format
