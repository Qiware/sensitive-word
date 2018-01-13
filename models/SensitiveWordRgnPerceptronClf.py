#coding=utf-8
import os

from sklearn import metrics
from sklearn import datasets
from sklearn.externals import joblib
from sklearn.linear_model import RidgeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

class SensitiveWordRgnPerceptronClf:
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
        fcache = "./data/model/perceptron.model"
        if os.path.isfile(fcache):
            print("Train model exists!")
            self.clf = joblib.load(fcache)
            return

        # 感知机: 一种最简单形式的前馈神经网络, 是一种二元线性分类器.
        self.clf = Perceptron(max_iter=50)
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
        # 加载训练数据
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
    clf = SensitiveWordRgnPerceptronClf('./data/train/')

    clf.Train()

    clf.Accuracy('./data/test/')
