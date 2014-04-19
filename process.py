import os
import gzip
import string

path = 'lingspam-mini600'

stop_words = []
subj_corpus = {}
body_corpus = {}

# Remove irrelevant words
# 	Returns 1 if irrevelant, otherwise 0
def filter(word):
	# Remove if standalone punctuation, special symbols and strings of numbers
	if (word in string.punctuation) or (word in string.digits):
		return 1
	# Remove if word is in list of stop words
	elif word in stop_words:
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
				if not filter(word):
					# Line contains subject
					if (line.startswith('Subject:')) and (word != 'Subject:'):
						# Increment or initialise item
						subj_corpus[word] = subj_corpus.get(word, 0) + 1
					# Line contains body
					else:
						body_corpus[word] = body_corpus.get(word, 0) + 1
	finally:
		f.close()

print sorted(subj_corpus.items(), key=lambda item: item[1], reverse=True)
print sorted(body_corpus.items(), key=lambda item: item[1], reverse=True)

