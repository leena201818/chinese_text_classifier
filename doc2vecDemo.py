# coding=utf-8
""" 
Created on 2016-02-17 @author: Zhan

输入：处理好的语料库
输出：word2vec模型
"""
import logging
import os.path
import sys
import multiprocessing
import codecs

from gensim.models import Word2Vec
from gensim.models import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from gensim import corpora

import getDictinary


class genSentence:
	myDict =  corpora.dictionary.Dictionary.load_from_text(u"字典.txt")
	def __iter__(self):
		dictSet =set(genSentence.myDict.values())
		for tName in getDictinary.dirIterator_tuple():
			wordsList_s = " ".join(codecs.open(tName[0],'rb','utf-8').readlines()).split()
			wordsList_st = [ item for item in wordsList_s if item in dictSet]
			documents = TaggedDocument(wordsList_st,[])
			yield documents
			#yield wordsList_st


if __name__ == "__main__":
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	doc2vecModel = Doc2Vec(genSentence(), size=100, window=8, min_count=5, workers=4)

	doc2vecModel.save('doc2vec.model')

