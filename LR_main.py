import numpy as np

def get_train_data():
    train_X = np.loadtxt(open('ML2020-PS2-dataset/train_set.csv','rb'),delimiter=",",skiprows=1,usecols=range(0,16))
    train_y = np.loadtxt(open('ML2020-PS2-dataset/train_set.csv','rb'),delimiter=",",skiprows=1,usecols=16)
    return train_X, train_y

def get_test_data():
    test_X = np.loadtxt(open('ML2020-PS2-dataset/test_set.csv','rb'),delimiter=",",skiprows=1,usecols=range(0,16))
    test_y = np.loadtxt(open('ML2020-PS2-dataset/test_set.csv','rb'),delimiter=",",skiprows=1,usecols=16)
    return test_X, test_y

def sigmoid(z):
    return 1/(1+np.exp(-z))

def get_accuracy(predicted_y, test_y):
    correct = 0
    for i in range(len(predicted_y)):
        if predicted_y[i]==test_y[i]:
            correct += 1
    return correct/len(predicted_y)

def get_microPrecision(TPList, FPList):
    TP=sum(TPList)
    FP=sum(FPList)
    microPrecision=TP/(TP+FP)
    return microPrecision

def get_microRecall(TPList, FNList):
    TP=sum(TPList)
    FN=sum(FNList)
    microRecall=TP/(TP+FN)
    return microRecall

def get_microF1(microPrecision, microRecall):
    microF1=2*microPrecision*microRecall/(microPrecision+microRecall)
    return microF1

def get_precision(TP, FP):
    precision=TP/(TP+FP)
    return precision

def get_recall(TP, FN):
    recall=TP/(TP+FN)
    return recall

def get_F1(precision, recall):
    F1=2*precision*recall/(precision+recall)
    return F1

def print_performance(accuracy, microPrecision, microRecall, microF1,macroPrecision,macroRecall,macroF1):
    print('accuracy: ',accuracy)
    print('micro Precision: ', microPrecision)
    print('micro Recall: ',microRecall)
    print('micro F1: ', microF1)
    print('macro Precision: ', macroPrecision)
    print('macro Recall: ',macroRecall)
    print('macro F1: ', macroF1)

class LR_clf:
    classNumber = 0 # 分哪一类
    beta = [] #参数，需要去拟合

    def __init__(self, classNumber):
        self.classNumber = classNumber

    def fit(self, train_X, train_y):
        # 将不属于该类的变成0，属于该类的变成1
        for i in range(len(train_y)):
            if train_y[i] != self.classNumber:
                train_y[i] = 0
            else:
                train_y[i] = 1
        # 初始值将beta置为全0
        self.beta = [0 for i in range(len(train_X[0]))]
        self.beta = SGD(self.beta, train_X, train_y)

    def predict(self, X):
        z = 0
        for i in range(len(X)):
            z += self.beta[i] * X[i]
        return sigmoid(z)

def SGD(beta, train_X, train_y):
    '''
    利用梯度下降法更新beta的值
    :param beta: beta的初始值
    :return: beta更新后的值
    '''
    alpha = 0.01 # learning rate
    iterations = 100 # 迭代300次计算

    for i in range(0, iterations):
        betax = []
        for j in range(0, len(train_X)):
            val = 0
            for k in range(len(train_X[0])):
                val += beta[k] * train_X[j][k]
            betax.append(val)

        gradient = [] # 计算每一个分量的梯度
        for j in range(len(beta)):
            ith_gradient = 0
            for k in range(len(train_X)):
                ith_gradient = ith_gradient - train_y[k] * train_X[k][j] + np.exp(betax[k]) * train_X[k][j] / (1 + np.exp(betax[k]))
            ith_gradient = ith_gradient / len(train_X)
            gradient.append(ith_gradient)

        # 更新每一个beta的值
        for j in range(len(beta)):
            beta[j] = beta[j] - gradient[j] * alpha
    print(beta)
    return beta

if __name__ == '__main__':
    train_X, train_y=get_train_data()
    test_X, test_y=get_test_data()

    b = np.ones(len(train_X))
    train_X = np.c_[train_X, b]
    b = np.ones(len(test_X))
    test_X = np.c_[test_X, b]

    # 储存26个分类器
    clfs = []
    for i in range(26):
        # 训练26个分类器
        clfs.append(LR_clf(i+1))
        clfs[i].fit(train_X.copy(), train_y.copy())

    # 对测试集进行预测
    predicted_y = []
    for X in test_X:
        max_p = 0
        predictClass = 0
        for i in range(len(clfs)):
            temp_p = clfs[i].predict(X)
            if temp_p > max_p:
                max_p = temp_p
                predictClass = i + 1
        predicted_y.append(predictClass)
    print(predicted_y)

    # predict result stored in predict_y
    # real result stored in test_y
    CLASSNUM = 26 # 类别数量

    TPList=[]
    FPList=[]
    TNList=[]
    FNList=[]

    for i in range(CLASSNUM):
        # 对每一类去计算各个评价指标值
        TP = 0
        FP = 0
        TN = 0
        FN = 0
        for j in range(len(test_y)):
            if predicted_y[j] == i+1 and test_y[j] == i+1:
                TP += 1
            elif predicted_y[j] == i+1 and test_y[j] != i+1:
                FP += 1
            elif predicted_y[j] != i+1 and test_y[j] == i+1:
                FN += 1
            elif predicted_y[j] != i+1 and test_y[j] != i+1:
                TN += 1
        TPList.append(TP)
        FPList.append(FP)
        TNList.append(TN)
        FNList.append(FN)


    # micro是用总的TP,FP,TN,FN求指标
    # macro是对各个分类器的指标求平均

    accuracy=get_accuracy(predicted_y, test_y)
    microPrecision=get_microPrecision(TPList, FPList)
    microRecall=get_microRecall(TPList, FNList)
    microF1=get_microF1(microPrecision, microRecall)
    macroPrecision = mean(precisionList)
    macroRecall = mean(recallList)
    macroF1 = mean(F1List)

    print_performance(accuracy, microPrecision, microRecall, microF1,macroPrecision,macroRecall,macroF1)