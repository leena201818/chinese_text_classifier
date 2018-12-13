# coding=utf-8
""" 
Created on 2016-02-17 @author: Zhan
word2vec SVM分类
输入：文本预料
输出：分类预测
"""
import os
import logging
import codecs
from pprint import PrettyPrinter

import gensim
from gensim import corpora, models, similarities
import matplotlib.pyplot as plt
from sklearn.svm import SVC 
# from sklearn.grid_search import GridSearchCV
from sklearn import datasets
from sklearn.externals import  joblib	
import numpy as np

import getDictinary
import segmentWord

# def classify(x,y):
# 	clf = GridSearchCV(SVC(random_state=42,max_iter=100),
# 		{'kernel':['linear','poly','rbf'],'C':[1,3,5,7,9,10]})

# 	clf.fit(x,y)
# 	print "Score", clf.score(x,y)
# 	PrettyPrinter().pprint(clf.grid_scores_)


if __name__ == "__main__":
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	logging.info(u"加载语料库")
	myCorpus = corpora.mmcorpus.MmCorpus('MyCorpus.mm')
	logging.info(u"加载字典")
	myDict =  corpora.dictionary.Dictionary.load_from_text(u"字典.txt")
	
	# 获取文档tfidf模型表示：并保存
	tfidf = models.TfidfModel(myCorpus) # step 1 -- initialize a model
	tfidf.save('tfidf.model')
	#  转为通用的numpy数组表示
	
	doc_x = gensim.matutils.corpus2csc(tfidf[myCorpus], num_terms=len(myDict.items()),
		num_docs=len(myCorpus)).transpose()
	doc_y = np.zeros(doc_x.shape[0])
	# logging.info(u"文本数量：" + str(doc_x.shape[0]))
	iLabel = 0;
	preClassName = ""
	label2Name = []
	for i,item in enumerate(getDictinary.dirIterator_tuple()):
		if i==0:
			doc_y[i] = 0
			# preClassName = unicode(item[1],'gbk')
			preClassName = item[1].decode('gbk').encode("utf-8")

			label2Name.append(preClassName)
		else:
			if item[1].decode('gbk').encode("utf-8") == preClassName:
				doc_y[i] = iLabel
			else:
				iLabel +=1
				doc_y[i] = iLabel
				preClassName = 	item[1].decode('gbk').encode("utf-8")
				label2Name.append(preClassName)
				# print preClassName
	
	# 保存分类名称对应的分类索引
	logging.info(u'分类索引和分类名称保存至文件')
	f = codecs.open('label2Name.txt','wb')
	for item in label2Name:
		f.write(item)
		f.write("\r\n")

	f.close()
	# 索引扰动			
	idx = np.arange(doc_x.shape[0])
	np.random.shuffle(idx)

	#	分割数据集 获取训练数据和测试数据
	doc_x_train = doc_x[idx[0:1000]]
	doc_y_train = doc_y[idx[0:1000]]

	doc_x_test = doc_x[idx[1000:]]
	doc_y_test = doc_y[idx[1000:]]

	# svc分类器构造函数
	'''
    class sklearn.svm.SVC(C=1.0, kernel='rbf', degree=3, gamma='auto', 
    	coef0=0.0, shrinking=True, probability=False, tol=0.001, 
    	cache_size=200, class_weight=None, verbose=False, 
    	max_iter=-1, decision_function_shape=None, random_state=None)
    
 	'''
 	knlType = 'linear'
 # 	CPar = 1.0
	# clf = SVC(C=CPar, kernel=knlType,)
	# clf.fit(doc_x_train,doc_y_train)
	# logging.info("C, kernel,Score: %f\t%s\t%s" % (CPar,knlType,clf.score(doc_x_test,doc_y_test)))


	CPar = 5.0
	clf = SVC(C=CPar, kernel=knlType,)
	clf.fit(doc_x_train,doc_y_train)
	logging.info("C:%f, kernel:%s,Score:%s" % (CPar,knlType,clf.score(doc_x_test,doc_y_test)))

	# CPar = 10.0
	# clf = SVC(C=CPar, kernel=knlType,)
	# clf.fit(doc_x_train,doc_y_train)
	# logging.info("C, kernel,Score: %f\t%s\t%s" % (CPar,knlType,clf.score(doc_x_test,doc_y_test)))


	# CPar = 1.0
	# knlType = "rbf"
	# clf = SVC(C=CPar, kernel=knlType,)
	# clf.fit(doc_x_train,doc_y_train)
	# logging.info("C, kernel,Score: %f\t%s\t%s" % (CPar,knlType,clf.score(doc_x_test,doc_y_test)))

	# CPar = 5.0
	# clf = SVC(C=CPar, kernel=knlType,)
	# clf.fit(doc_x_train,doc_y_train)
	# logging.info("C, kernel,Score: %f\t%s\t%s" % (CPar,knlType,clf.score(doc_x_test,doc_y_test)))

	# CPar =10.0
	# clf = SVC(C=CPar, kernel=knlType,)
	# clf.fit(doc_x_train,doc_y_train)
	# logging.info("C, kernel,Score: %f\t%s\t%s" % (CPar,knlType,clf.score(doc_x_test,doc_y_test)))

	# CPar = 1.0
	# knlType = "poly"
	# clf = SVC(C=CPar, kernel=knlType,)
	# clf.fit(doc_x_train,doc_y_train)
	# logging.info("C, kernel,Score: %f\t%s\t%s" % (CPar,knlType,clf.score(doc_x_test,doc_y_test)))


	# CPar = 5.0
	# knlType = "poly"
	# clf = SVC(C=CPar, kernel=knlType,)
	# clf.fit(doc_x_train,doc_y_train)
	# logging.info("C, kernel,Score: %f\t%s\t%s" % (CPar,knlType,clf.score(doc_x_test,doc_y_test)))


	# CPar = 10.0
	# knlType = "poly"
	# clf = SVC(C=CPar, kernel=knlType,)
	# clf.fit(doc_x_train,doc_y_train)
	# logging.info("C, kernel,Score: %f\t%s\t%s" % (CPar,knlType,clf.score(doc_x_test,doc_y_test)))

	logging.info(u"加载文件映射、文件名称及类别")
	f = open("index2File.txt",'r')
	index2File = []
	for strLine in f.readlines():
		strList = strLine.split()
		index2File.append((strList[1],strList[2]))

	f.close()


	tNum = 0.00
	for i,item in enumerate(doc_x_test):
		realLabel = index2File[idx[i+1000]][1]
		predictLabe = label2Name[(int)(clf.predict(doc_x_test[i])[0])]
		if realLabel == predictLabe:
			tNum +=1;
		else:
			info  = "the %6d elemente, filename: %s,real class is %s, predicte classs: %s" % (idx[i+1000],
			index2File[idx[i+1000]][0],realLabel,predictLabe)
			logging.info(info)
			info  = "the %6d elemente, filename: %s,real class is %s" %(idx[i+1000],index2File[idx[i+1000]][0],index2File[idx[i+1000]][1],)
	
	# info = u"测试集上预测准确率：%f".decode("utf")
	accuracy = 	tNum/doc_x_test.shape[0]
	logging.info("predict accuracy on test dataset: %f%%" % accuracy)

	# 保存模型
	logging.info(u"将支持向量机分类器保存至文件，以备下次使用")
	joblib.dump(clf,'svmTxt.pkl')








			

















