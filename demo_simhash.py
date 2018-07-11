from simhash import Simhash
import pandas as pd

def get_xls_data():
    # 获取数据
    data = pd.read_excel("./data/排重案例样本0706.xlsx", names=["content1", "content2"], sheetname=[0])
    content_ls_1 = [(x, y) for x, y in enumerate(data[0]["content1"]) if y]
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
        distance = (Simhash(x[0]).distance(Simhash(x[1])))
        if distance <= 25:
            print(distance)
            print("相似文本为:" + x[0] + "    |||||   " + x[1])
            print("==================")
            similarity_length = similarity_length + 1

    print("相似语料长度:" + str(similarity_length))
    print("相似度识别成功率:%s" % (similarity_length / len(content_ls)) * 100 + "%")


if __name__ == '__main__':
    get_xls_data()