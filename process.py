import os
import gzip
import string

path = 'lingspam-mini600'
subj = []
body = []

# subj_corpus = {}
# body_corpus = {}
#
# for file in os.listdir(path):
# 	f = gzip.open(os.path.join(path, file), 'rb')
# 	try:
# 		for line in f:
# 			for word in line.split():
# 				# Line contains subject
# 				if line.startswith('Subject:'):
#					# Increment or initialise item
# 					subj_corpus[word] = subj_corpus(word, 0) + 1
# 				# Line contains body
# 				else:
# 					body_corpus[word] = body_corpus(word, 0) + 1
# 	finally:
# 		f.close()
#
# print subj_corpus
# print body_corpus

# Count the frequency of words in a list
def word_freq(list):
    corpus = {}
    for i in range(len(list)):
        corpus[list[i]] = list.count(list[i])
    return sorted(corpus.items(), key=lambda item: item[1], reverse=True)


# Remove irrelevant words
# 	Returns 1 if irrevelant, otherwise 0
def filter(word):
	# Remove standalone punctuation, special symbols and strings of numbers
	if (word in string.punctuation) or (word in string.digits):
		return 1
	else:
		return 0

# Append words to either subject or body list
for file in os.listdir(path):
	f = gzip.open(os.path.join(path, file), 'rb')
	for line in f:
		for word in line.split():
			# Check if word is to be kept
			if not filter(word):
				# Line contains subject
				if line.startswith('Subject:') and (word != 'Subject:'):
					subj.append(word)
				# Line contains body
				else:
					body.append(word)

print word_freq(subj)
print word_freq(body)







