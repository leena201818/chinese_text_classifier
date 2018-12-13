# coding=utf-8
""" 
Created on 2016-05-02 @author: Zhan
 SVM分类
输入：要分类文本
输出：分类标签
"""
import logging

import gensim
from gensim import corpora, models, similarities
import matplotlib.pyplot as plt
from sklearn.svm import SVC 
from sklearn import datasets
from sklearn.externals import  joblib	
import numpy as np

import segmentWord

	
if __name__ == "__main__":
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	
	# 加载类别名称索引
	logging.info(u"加载类别名称索引")
	label2Name = []
	f = open('label2Name.txt')
	for item in f.readlines():
		label2Name.append(item)
	f.close()
	

	# step1 加载词典、tfidf模型、语料
	logging.info(u"加载字典")
	myDict =  corpora.dictionary.Dictionary.load_from_text(u"字典.txt")
	logging.info(u"加载文档tfidf模型")
	tfidf = models.TfidfModel.load('tfidf.model')
	logging.info(u"加载语料库")
	clf = joblib.load("svmTxt.pkl")
	# step2  读取文件、分词
	stopwords = {}.fromkeys([line.rstrip() for line in open(u"停用词_精简.txt")])
	#fName = u'D:\\zhan_study\\NLP\\0098.txt'
	fName = u'D:\\Python27_NLP\\语料库\\动物\\BaiduSpiderAnimal\\0098.txt'
	words = segmentWord.cut_file(fName,stopwords)

	# step3  获取文档向量表示
	tmpDoc = myDict.doc2bow(words)

	#setp4 获取文档的tfidf表示
	tmpDoc_tfidf = gensim.matutils.corpus2csc(tfidf[[tmpDoc]], num_terms=len(myDict.items()))
	# step5 预测分类标签

	predictLabel = label2Name[(int)(clf.predict(tmpDoc_tfidf.transpose())[0])]
	info  = "predicte classs: %s" % predictLabel
	logging.info(info)
