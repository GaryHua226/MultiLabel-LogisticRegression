from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
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

def get_accuracy(predicted_y):
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


if __name__ == '__main__':
    train_X, train_y=get_train_data()
    test_X, test_y=get_test_data()


    # micro是用总的TP,FP,TN,FN求指标
    # macro是对各个分类器的指标求平均
