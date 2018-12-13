# coding=utf-8
""" 
Created on 2016-02-17 @author: Zhan
对原始文档进行分词
输入：文件夹中对应0001.txt-0100.txt
输出：每个文件对应分词结果保存在相应的文件中
"""

import sys
import re
import codecs
import os
import shutil
import logging


import jieba
import jieba.analyse

#全局日志对象
logger = logging.getLogger('mylogger')
#recursively Read file and cut
def recursive_file_cut(dir,stopwords):

    #去除标点符号
    # pointwords = {}.fromkeys(['，', '、', '[', ']', '（', '）', '：',
    #     '、', '。', '@', '’', '%', '《', '》', '“', '”', '.', '；',
    #     '′', '°', '″', '-', ',', '！', '？','～', '\'', '\"', ':',
    #     '(', ')', '【', '】', '~', '/', ';', '→', '\\', '·', '℃'])
    
    logger.info(dir)
    for subDir in os.listdir(dir):
        fsubDir = os.path.join(dir,subDir)
        logger.info(fsubDir )
        if os.path.isdir(fsubDir):
            label = subDir # 标签

            para = {'stopwords':stopwords,'segList':[]}
            # logger.info(fsubDir)
            os.path.walk(fsubDir,cut_file_label,para)
            tDir = os.path.join(os.getcwd(),subDir)
            if os.path.exists(tDir):
                for f in os.listdir(tDir):
                    os.remove(os.path.join(tDir,f))
                os.rmdir(tDir)
            os.mkdir(tDir)
            logger.info(os.path.join(dir,subDir))
            for i,words in enumerate(para['segList']):
                if len(words)<5:# 去掉短文档,这里去掉不足5个词语的文档
                    continue
                resName = ("%06d" % i )+ ".txt" 
                logger.info(os.path.join(tDir, resName))
                result = codecs.open(os.path.join(tDir, resName), 'w', 'utf-8')
                result.write(" ".join(words))
                result.write("\r\n")
                result.close()
                logger.info(u"分词已保存至文件 " + os.path.join(tDir, resName))

def cut_file_label(para,curDir,subDirs):
    for subDir in subDirs:
        fname = os.path.join(curDir,subDir)
        if os.path.isfile(fname):
            segList = cut_file(fname,para['stopwords'])
            para['segList'].append(segList)
        elif  os.path.isdir(fname):
            logger.info(fname)



# 对单个文件分词，返回分词后的单词列表；
def cut_file(fileName,stopwords):
    # global stopwords
    # print stopwords
    source = open(fileName, 'r')
    line = source.readline()
    output = []
    while line!="":
        line = line.rstrip('\n')
        seglist = jieba.cut(line,cut_all=False)  #精确模式
        output.extend([item for item in seglist if item.encode("utf8") not in stopwords])         
        #取下一行
        line = source.readline()
    else:
        # logger.info("%s is completed!" %fileName)
        logger.info(u"%s 文件处理完毕!" %fileName)
        source.close()

    return output

#Run function
if __name__ == '__main__':
    # 日志设置
    logger.setLevel(logging.INFO)

    # 创建一个handler，用于写入日志文件  
    fh = logging.FileHandler('test.log',mode='w')  
    fh.setLevel(logging.INFO)  

    # 再创建一个handler，用于输出到控制台  
    ch = logging.StreamHandler()  
    ch.setLevel(logging.INFO)  

    #  日志格式
    formatter = logging.Formatter('%(asctime)s : %(levelname)s : %(message)s')  
    fh.setFormatter(formatter)  
    ch.setFormatter(formatter)  

    # 给logger添加handler  
    logger.addHandler(fh)  
    logger.addHandler(ch)  

    #导入自定义词典
    # logging.level =
    jieba.load_userdict("user_dict.txt") 
    logger.info(u"加载停用词词典 停用词_精简.txt")
    stopwords = {}.fromkeys([line.rstrip() for line in open(u"停用词_精简.txt")])
    # 命令行中文内参数一定要转为unicode编码
    # dirName = unicode(sys.argv[1],'gbk')
    # logger.info(dirName)
    # recursive_file_cut(dirName)
    #recursive_file_cut(u'D:\软件工具\python\python包\gensim_jieba\语料库',stopwords)
    recursive_file_cut(u'D:\Python27_NLP\语料库',stopwords)
    # recursive_file_cut(dirName)
