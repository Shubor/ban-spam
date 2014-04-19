import os
import re
import gzip
from string import digits
from string import punctuation
from collections import Counter

path = 'lingspam-mini600'

# Select only top N words
N = 200

subj_corpus = Counter()
body_corpus = Counter()

# Check if word is not irrelevant
# 	Returns 1 if irrevelant, otherwise 0
def clean(word):
	# Remove if punctuation, special symbols, number or word is in stop words
	if word in punctuation or word.isdigit() or word in stop_words:
		return 1
	# Remove strings that contain digits e.g. 'qwoie192kjwe' or '12312-123'
	# 	or contain punctuation e.g. "n't"
	elif re.compile('\d').search(word) or re.compile('\W').search(word):
		return 1
	else:
		return 0

# Read in stop words from file
stop_words = []
with open('english.stop', 'rU') as stop_file:
	stop_words = [line.rstrip('\n') for line in stop_file]

# Append words to either subject or body list
curr_file_words = []
for file in os.listdir(path):
	f = gzip.open(os.path.join(path, file), 'rb')
	try:
		for line in f:
			# Replace punctuation with space
			# l = re.sub('[%s]' % re.escape(punctuation), ' ', line)
			for word in line.split():
				# Check if word is to be kept
				if not clean(word) and word not in curr_file_words:
					curr_file_words.append(word)
					# Line contains subject
					if line.startswith('Subject:'):
						if word != 'Subject:':
							subj_corpus[word] += 1
					# Line contains body
					else:
						body_corpus[word] += 1
	finally:
		f.close()
		curr_file_words[:] = []

print(subj_corpus.most_common(N))
print(body_corpus.most_common(N))
