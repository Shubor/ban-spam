def printTable( x ):
	# x and y are lists of top 100 in sorted order
 	for i in range(34):
 		if i == 32 or i == 33:
 			break
 		print("{} & {} & {} & {} & {} & {} & {} & {} & {} \\\\".format( i+1, x[i][0], x[i][1], i+35, x[i+34][0], x[i+34][1], i+69, x[i+68][0], x[i+68][1] ))
 	print("{} & {} & {} & {} & {} & {} &  &  &  \\\\".format( 33, x[32][0], x[32][1], 67, x[66][0], x[66][1] ))
 	print("{} & {} & {} & {} & {} & {} &  &  &  \\\\".format( 34, x[33][0], x[33][1], 68, x[67][0], x[67][1] ))

prinTable( body_corpus.most_common(100) )