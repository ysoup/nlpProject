# coding:utf-8

import jieba
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity, pairwise_distances
import math

jieba.suggest_freq('以太坊', True)
jieba.suggest_freq('区块链', True)
jieba.suggest_freq('数字货币', True)
jieba.suggest_freq('将于', True)
jieba.suggest_freq('人人网', True)
jieba.suggest_freq('比特币', True)
jieba.suggest_freq('北上广', True)
jieba.suggest_freq('大数据', True)
jieba.suggest_freq('云计算', True)
jieba.suggest_freq('公有链', True)
# 引用停用词
stpwrdpath = "./data/stop_words.txt"
stpwrd_dic = open(stpwrdpath, 'rb')
stpwrd_content = stpwrd_dic.read()
# 将停用词表转换为list
stpwrdlst = stpwrd_content.splitlines()
stpwrd_dic.close()
# vector = TfidfVectorizer(stop_words=stpwrdlst)


def get_xls_data():
    # 获取数据
    data = pd.read_excel("./data/排重案例样本0706.xlsx", names=["content1", "content2"], sheetname=[0])
    content_ls_1 = [(x, y) for x, y in enumerate(data[0]["content1"]) if y]
    #print(content_ls_1)
    content_ls_2 = [(x, y) for x, y in enumerate(data[0]["content2"]) if y]
    content_ls = []
    for x in content_ls_1:
        for y in content_ls_2:
            if x[0] == y[0]:
                content_ls.append((x[1], y[1]))

    # 数据分词
    print("语料长度:" + str(len(content_ls)))
    similarity_length = 0
    for x in content_ls:
        # print([get_jieba_doc(x[0]), get_jieba_doc(x[1])])
        vectorizer = CountVectorizer()
        transformer = TfidfTransformer()
        tfidf = transformer.fit_transform(vectorizer.fit_transform([get_jieba_doc(x[0]), get_jieba_doc(x[1])]))
        # print(cosine_similarity(tfidf))
        # print("======================================")
        vector = TfidfVectorizer(max_df=10, min_df=1)
        tfidf = vector.fit_transform([get_jieba_doc(x[0]), get_jieba_doc(x[1])])
        new_cosine_similarity = cosine_similarity(tfidf).tolist()
        if new_cosine_similarity[0][1] > 0.7:
            print(cosine_similarity(tfidf))
            print("相似文本为:" + x[0]+"    |||||   " + x[1])
            print("==================")
            similarity_length = similarity_length + 1

    print("相似语料长度:" + str(similarity_length))
    print("相似度识别成功率:%s" % (similarity_length/len(content_ls))*100 + "%")


def get_jieba_doc(document):
    document_cut = jieba.cut(document)
    try:
        return " ".join(document_cut)
    except Exception as e:
        print(e.message)


# 计算向量夹角余弦
def VectorCosine(x, y):
    vc = []
    for i in range(1, len(x)-2):
        xc1 = x[i] - x[i-1]
        xc2 = x[i+1] - x[i]
        yc1 = y[i] - y[i-1]
        yc2 = y[i+1] - y[i]
        vc.append((xc1*xc2+yc1*yc2)/(math.sqrt(xc1**2+yc1**2)*math.sqrt(xc2**2+yc2**2)))

    return vc


if __name__ == '__main__':
    get_xls_data()




