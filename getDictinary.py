# coding=utf-8
""" 
Created on 2016-02-17 @author: Zhan
得到词典
输入：分词后的文件
输出：corpus和dictionnary
"""
import os
import logging
import codecs

from gensim import corpora
from six import iteritems


# class dirIterator:
# 	def __iter__(self):
# 		for tName in os.listdir(os.getcwd()):
# 			fName = os.path.join(os.getcwd(),tName)
# 			if os.path.isdir(fName):
# 				for file in os.listdir(fName):
# 					yield os.path.join(fName,file)

class dirIterator_tuple:
	def __iter__(self):
		for tName in os.listdir(os.getcwd()):
			fName = os.path.join(os.getcwd(),tName)
			if os.path.isdir(fName):
				for file in os.listdir(fName):
					# print os.path.join(fName,file),tName
					yield (os.path.join(fName,file),tName)					

class genDoc:
	def __iter__(self):
		for fName in dirIterator_tuple():
					yield " ".join(open(fName[0]).readlines()).split()

if __name__ == "__main__":
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	dictionary = corpora.Dictionary(genDoc())	

	# 停用词 	
	# stopwords = []
	stopwords = [line.rstrip() for line in codecs.open("stopWords.txt", 'rb', 'utf-8')]
	stopwords_ids = [dictionary.token2id[stopword] for stopword in stopwords if stopword in dictionary.token2id]
	logging.info(u"停用词长度：%d" % len(stopwords))
	logging.info(u"停用词ID列表长度：%d" % len(stopwords_ids))
	# stopwords = []

	# 出现次数少的词
	minorWords_ids = [tokenid for tokenid, docfreq in iteritems(dictionary.dfs) if docfreq <= 1]
	logging.info(u"文档频率少的词：%d"%len(minorWords_ids))
	#对字典过滤
	dictionary.filter_tokens(stopwords_ids + minorWords_ids)
	dictionary.compactify()  # remove gaps in id sequence after words that were removed
	logging.info(u"字典长度：%d"  % len(dictionary.items()))

	dictionary.save_as_text(os.path.join(os.getcwd(),u"字典.txt"))







