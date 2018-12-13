# coding=utf-8
""" 
Created on 2016-02-17 @author: Zhan
word2vec 演示生成词云
输入：
输出：
"""
import os
import logging

import numpy as np
import matplotlib.pyplot as plt
from scipy.misc import imread
from wordcloud import WordCloud,ImageColorGenerator
import jieba


if __name__ == "__main__":
	logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

	# 单个文件词云
	text_from_file_with_apath = open(os.path.join(os.getcwd(),u'人物/000053.txt'),'r').read()
	wordlist_after_jieba = jieba.cut(text_from_file_with_apath, cut_all = True)
	wl_space_split = " ".join(wordlist_after_jieba)
	
	# alice_coloring = np.array(Image.open(path.join(d, "girl_colored.png")))
	# alice_coloring = np.array(imread('girl_colored.png'))
	alice_coloring = imread('love.jpg')


	wc = WordCloud( mask = alice_coloring, 
					background_color="white",
					font_path = r'C:/Windows/Fonts/msyhbd.ttf')

	# 同类多文件词云
	# wl_space_split = ""
	# dirName = os.path.join(os.getcwd(),u'景区')
	# for fName in os.listdir(dirName):
	# 	text_from_file_with_apath = open(os.path.join(dirName,fName),'r').read()
	# 	wordlist_after_jieba = jieba.cut(text_from_file_with_apath, cut_all = True)
	# 	wl_space_split += " ".join(wordlist_after_jieba)
	
	
	
	# 从字典生词云
	# dict = {"java":50,"c++":40,"c#":40,"php":40,u"web网站":40,u"手机APP":30,
	# "python":40,"matlab":30,u"simulink仿真":30,u"编程作业":30,u"毕业设计":30,u"留学生作业":30,"asp.net":30}
	# my_wordcloud = wc.generate_from_frequencies(dict)

	my_wordcloud = wc.generate(wl_space_split)

	plt.imshow(wc)
	plt.axis("off")
	plt.figure()

	image_colors = ImageColorGenerator(alice_coloring)
	wc.recolor(color_func = image_colors)
	plt.imshow(wc, interpolation="bilinear")
	plt.axis("off")
	plt.figure()

	plt.imshow(alice_coloring, cmap=plt.cm.gray, interpolation="bilinear")
	plt.axis("off")
	plt.show()
