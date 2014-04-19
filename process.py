import os
import gzip
from string import punctuation
from collections import Counter

path = 'lingspam-mini600'

# Select only top N words
N = 200

stop_words = []

subj_corpus = Counter()
body_corpus = Counter()

# Remove irrelevant words
# 	Returns 1 if irrevelant, otherwise 0
def clean(word):
	# Remove if punctuation, special symbols, number or word is in stop words
	if word in punctuation or word.isdigit() or word in stop_words:
		return 1
	else:
		return 0

# Read in stop words from file
with open('english.stop', 'rU') as stop_file:
	stop_words = [line.rstrip('\n') for line in stop_file]

# Append words to either subject or body list
for file in os.listdir(path):
	f = gzip.open(os.path.join(path, file), 'rb')
	try:
		for line in f:
			for word in line.split():
				# Check if word is to be kept
				if not clean(word):
					# Line contains subject
					if (line.startswith('Subject:')) and (word != 'Subject:'):
						# Increment or initialise item
						subj_corpus[word] += 1
					# Line contains body
					else:
						body_corpus[word] += 1
	finally:
		f.close()

print subj_corpus.most_common(N)
print body_corpus.most_common(N)
