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
from sklearn.cluster import AffinityPropagation
from sklearn import metrics

import segmentWord
import getDictinary
	
class getIndx2FileIter:
	def __iter__(self):
		for line in codecs.open('index2File.txt','rb','utf-8').readlines():
			yield line.split()


if __name__ == "__main__":
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
	
	#加载文件映射、文件名称及类别
	logging.info(u"加载文件映射、文件名称及类别")



	f = open("index2File.txt",'r')	
	index2File = []
	for strLine in f.readlines():
		strList = strLine.split()
		index2File.append((strList[1],strList[2]))	
	f.close()

	
	# step1 加载tfidf模型、语料库：
	# logging.info(u"加载字典")
	myDict =  corpora.dictionary.Dictionary.load_from_text(u"字典.txt")
	logging.info(u"加载文档tfidf模型")
	tfidf = models.TfidfModel.load('tfidf.model')
	logging.info(u"加载语料库")
	myCorpus = corpora.mmcorpus.MmCorpus('MyCorpus.mm')


	#setp2  获取文档的tfidf表示和每个文档的类别
	tmpDoc_tfidf = tfidf[myCorpus]
	# tmpDoc_tfidf = gensim.matutils.corpus2csc(tfidf[myCorpus]).transpose()

	# doc_y = np.zeros(tmpDoc_tfidf.shape[0])
	doc_y = np.zeros(len(tmpDoc_tfidf))

	iLabel = 0;
	preClassName = ""
	for i,item in enumerate(getDictinary.dirIterator_tuple()):
		if i==0:
			doc_y[i] = 0
			preClassName = item[1].decode('gbk').encode("utf-8")
		else:
			if item[1].decode('gbk').encode("utf-8") == preClassName:
				doc_y[i] = iLabel
			else:
				iLabel +=1
				doc_y[i] = iLabel
				preClassName = 	item[1].decode('gbk').encode("utf-8")

	# step3 执行affinity Propagation  聚类
	'''
	AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15, copy=True, preference=None, 
	affinity='euclidean', verbose=False)
	'''
	# pref = np.ones(len(tmpDoc_tfidf))
	# pref = pref/10.0
	# pref[50] = 0.60
	# pref[550] = 0.55
	# pref[850] = 0.755
	# pref[1400] = 0.53
	# pref =  np.random.normal(size=(len(tmpDoc_tfidf)))

	# af = AffinityPropagation(max_iter=1000,preference = pref).fit(tmpDoc_tfidf)

	logging.info(u'计算相似性矩阵')
	index = similarities.docsim.SparseMatrixSimilarity(tmpDoc_tfidf,num_features = len(myDict.items()))
	sims = index[tmpDoc_tfidf]
	af = AffinityPropagation(convergence_iter=100,max_iter=1000).fit(sims)
	cluster_centers_indices = af.cluster_centers_indices_
	labels = af.labels_

	n_clusters_ = len(cluster_centers_indices)


	logging.info('Estimated number of clusters: %d' % n_clusters_)
	logging.info("Homogeneity: %0.3f" % metrics.homogeneity_score(doc_y, labels))
	logging.info("Completeness: %0.3f" % metrics.completeness_score(doc_y, labels))
	logging.info("V-measure: %0.3f" % metrics.v_measure_score(doc_y, labels))
	logging.info("Adjusted Rand Index: %0.3f" % metrics.adjusted_rand_score(doc_y, labels))
	logging.info("Adjusted Mutual Information: %0.3f" % metrics.adjusted_mutual_info_score(doc_y, labels))
	# logging.info("Silhouette Coefficient: %0.3f" % metrics.silhouette_score(tmpDoc_tfidf, labels, metric='sqeuclidean'))

	for iLabel in range(n_clusters_):
		logging.info("the %dth cluster contains the following files:" % iLabel)
		tmpList = [index2File[i] for i, item in enumerate(labels) if item == iLabel]
		for item in tmpList:
			logging.info(item[0])


