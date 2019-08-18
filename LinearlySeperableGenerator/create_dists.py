import pickle
from nltk.corpus import brown
from nltk.probability import ConditionalFreqDist
from nltk.tokenize import word_tokenize
from nltk import data

def seperate( x ):
	ngr = len(x)
	ret = []
	n_1_gr = []
	for i in range(ngr-1):
		n_1_gr.append(x[i])
		ret.append( tuple(n_1_gr) )
	ret.append( x[ngr-1] )
	return tuple(ret)


# def pickle_trigram_brown( name ):
# 	trg = list ( ngrams( brown.words() , n=3 ) )
# 	trigrams_dist = ConditionalFreqDist( ( (x[0],x[1]), x[2] ) for x in trg )
# 	f = open(name, "wb")
# 	pickle.dump( trigrams_dist, f )

def pickle_ngram_brown( name, n ):
	trg = list ( ngrams( brown.words() , n=n ) )
	ngrams_dist = ConditionalFreqDist(  seperate(x) for x in trg )
	f = open(name, "wb")
	pickle.dump( ngrams_dist, f )
	f.close()

def pickle_ngram_tamil( corpus, name, n ):
	x = data.load( corpus )
	tokens = word_tokenize( x )
	trg = list( ngrams( tokens , n=n ) )
	ngrams_dist = ConditionalFreqDist(  seperate(x) for x in trg )
	f = open(name, "wb" )
	pickle.dump( ngrams_dist, f )
	f.close()