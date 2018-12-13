# coding=utf-8
""" 
Created on 2016-02-17 @author: Zhan
生成语料库
输入：分词后的文件
输出：corpus和dictionnary
"""
import os
import logging
import codecs
from gensim import corpora
from six import iteritems
import getDictinary


class MyCorpus:
	myDict =  corpora.dictionary.Dictionary.load_from_text(u"字典.txt")
	def __iter__(self):
		for tName in getDictinary.dirIterator_tuple():
			yield MyCorpus.myDict.doc2bow(" ".join(open(tName[0]).readlines()).split())

if __name__ == "__main__":
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	logging.info(u"加载字典模型")
	corpus_memory = MyCorpus()
	corpora.mmcorpus.MmCorpus.serialize('MyCorpus.mm', corpus_memory)

	logging.info(u"保存文件名映射关系")
	f = open("index2File.txt","w")
	for i,tFile in enumerate(getDictinary.dirIterator_tuple()):
		f.write("%6d \t %s\t %s" % (i,tFile[0],tFile[1]))
		f.write(os.linesep)

	f.close()
	# load_from_text(u"字典.txt")





