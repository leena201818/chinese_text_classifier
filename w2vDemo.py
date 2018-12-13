# coding=utf-8
""" 
Created on 2016-02-17 @author: Zhan
word2vec 演示
输入：
输出：
"""
import logging
import os.path
import sys
import multiprocessing
import codecs

from gensim.models import Word2Vec

if __name__ == "__main__":
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	logging.info(u"加载woed2vec模型")
	
	wordModel = Word2Vec.load('word2vec.model')
	
	logging.info(u'中国的向量表示:')
	for item in wordModel.wv[u'中国']:
		logging.info(item) 

	logging.info(u'寻找最相似的词:')
	for item in wordModel.most_similar(u"中国"):
		logging.info(item[0])
		logging.info(item[1])

	# logging.info(u'寻找最相似的词:')
	# for item in wordModel.most_similar(positive=[u"女人",u'国王'],negative=[u'男人']):
	# 	logging.info(item[0])
	# 	logging.info(item[1])

	logging.info(u'寻找最相似的词:')
	for item in wordModel.most_similar_cosmul(positive=[u"女人",u'国王'],negative=[u'男人']):
	 	logging.info(item[0])
	 	logging.info(item[1])		

	
	logging.info(u'寻找不合群的:')
	logging.info(wordModel.wv.doesnt_match(u"中国 美国 日本 马来熊".split()))

	# logging.info(u'寻找不合群的:')
	# logging.info(wordModel.wv.doesnt_match(u"铁路 公路 汽车 日本".split()))


	logging.info(u'寻找不合群的:')
	logging.info(wordModel.wv.doesnt_match(u"长颈鹿  仙人掌  棕熊 马来熊  牧羊犬".split()))


	logging.info(u'计算句子存在的可能性:')
	#logging.info(wordModel.score(u"小日本 就是 该 灭亡"))

