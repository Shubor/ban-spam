ban-spam
========

- Open file
    - Read file
        - Make dictionary of ‘subject’ words
        - Make dictionary of ‘body’ words
    - Return N=200 most frequent in 'body'
- Close file

- Calculate tf - idf
    + t_k := the number of times the word occurs in the document d_k
    + d_k := current document
    + |Tr| := is the size of this collection (total number of documents)
    + Tr(t_k) := the document frequency of t_k
    + #(t_k,d_j) := the number of times the word t_k occurs in the document d_j
    + tf-idf := #(t_k,d_j) * log( |Tr|/#Tr(t_k) )
- Normalise tf - idf b/w 0-1 using cosine normalisation