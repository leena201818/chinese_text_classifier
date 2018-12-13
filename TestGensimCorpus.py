import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

from gensim import corpora

documents = ["Human machine interface for lab abc computer applications",
             "A survey of user opinion of computer system response time",
             "The EPS user interface management system",
             "System and human system engineering testing of EPS",
             "Relation of user perceived response time to error measurement",
             "The generation of random binary unordered trees",
             "The intersection graph of paths in trees",
             "Graph minors IV Widths of trees and well quasi ordering",
             "Graph minors A survey"]
# remove common words and tokenize

stoplist = set('for a of the and to in'.split())

texts =[ [ word for word in document.lower().split() if word not in stoplist]  for document in documents]

print(texts)

#remove words that appear only once
from collections import defaultdict
frequency = defaultdict(int)
for text in texts:
	for token in text:
		frequency[token] += 1

texts = [ [token for token in text if frequency[token] > 1] for text in texts]

from pprint import pprint # pretty-printer
pprint(texts)

dictionary = corpora.Dictionary(texts)
dictionary.save('D:\\Python27_NLP\\mylearn\\deerwester.dict')
print(dictionary)

print(dictionary.token2id)

new_doc = "Human computer comPuter interaction"
new_vec = dictionary.doc2bow(new_doc.lower().split())
print(new_vec) #the word "interaction" does not apppear in the dictionary and is ignored

corpus = [ dictionary.doc2bow(doc) for doc in texts]
corpora.MmCorpus.serialize('D:\\Python27_NLP\\mylearn\\deerwester.mm',corpus) # store to disk,for later use
pprint(corpus)


class MyCorpus(object):
	def __iter__(self):
		for line in open('D:\\Python27_NLP\\mylearn\\mycorpus.txt'):
			#assume there is one document per line,tokens separated by whitespace			
			yield dictionary.doc2bow(line.lower().split())

corpus_memory_friendly = MyCorpus()	#does not load the corpus into memory!
print(corpus_memory_friendly)

for vector in corpus_memory_friendly:	#load one vector into memory at a time
	pprint(vector)


from six import iteritems
# collect statistics about all tokens
dictionary = corpora.Dictionary(line.lower().split() for line in open('D:\\Python27_NLP\\mylearn\\mycorpus.txt'))
#remove stop words and words that appear only once
stop_ids = [dictionary.token2id[stopword] for stopword in stoplist if stopword in dictionary.token2id]
#pprint(dictionary.token2id)
ones_ids = [ tokenid for tokenid,docfreq in iteritems(dictionary.dfs) if docfreq == 1]
dictionary.filter_tokens(stop_ids + ones_ids)  #remove stop words and words that appear only once
dictionary.compactify()	#remove gaps in id sequence after words that were removed
