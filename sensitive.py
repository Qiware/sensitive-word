#coding=utf-8
import os

from sklearn import metrics
from sklearn import datasets
from sklearn.externals import joblib
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

class SensitiveClf:
    def __init__(self, path):
        self.clf = None
        # 加载训练数据
        self.ds = datasets.load_files(path);
        if self.ds is None:
            print("Load dataset failed! path:%s" % (path))
            return

        # 矢量化训练集: 使用TF-IDF值矢量化训练集.
        # TFIDFi,j = TFi,j*IDFi,j
        # TF-IDF: Term Frequency & InversDocument Frequency
        self.vec = TfidfVectorizer(sublinear_tf=True, max_df=0.5, stop_words='english')
        if self.vec is None:
            print("Init TF-IDF failed!")
            return

        self.X = self.vec.fit_transform(self.ds.data)
        if self.X is None:
            print("Get X failed!")
            return

        self.Y = self.ds.target
        if self.Y is None:
            print("Get Y failed!")
            return

        return

    # 进行数据训练
    def Train(self):
        fcache = "./data/model/train.mod"
        if os.path.isfile(fcache):
            print("Train model exists!")
            self.clf = joblib.load(fcache)
            return

        #随机森林就是通过集成学习的思想将多棵树集成的一种算法，它的基本单元是决策树。
        #每棵决策树都是一个分类器，那么对于一个输入样本，N棵树会有N个分类结果。
        #而随机森林集成了所有的分类投票结果，将投票次数最多的类别指定为最终的输出
        #self.clf = RandomForestClassifier(n_estimators=100)
        self.clf = RidgeClassifier(tol=1e-2, solver="sag")
        self.clf.fit(self.X, self.Y)

        # 将训练模型存储到磁盘 - 下次启动时大幅减少训练时间
        if not os.path.exists("./data/model"):
            os.makedirs("./data/model")

        print("Dump train model!")

        joblib.dump(self.clf, fcache)
        return

    # 进行预测处理
    def Predict(self, sentence):
        print("Didn't implement predict!")
        return

    # 校验测试数据
    def Accuracy(self, path):
        # 加载测试数据
        ds = datasets.load_files(path);
        if ds is None:
            print("Load dataset failed! path:%s" % (path))
            return

        test_X = self.vec.transform(ds.data)
        test_Y = ds.target

        p = self.clf.predict(test_X)
        score = metrics.accuracy_score(test_Y, p)

        print("精确度: %0.3f" % score)
        return

if __name__ == "__main__":
    # 创建敏感词分类器
    clf = SensitiveClf('./data/train/')

    clf.Train()

    clf.Accuracy('./data/test/')
