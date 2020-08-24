#coding=utf-8
#使用CART对手写数字分类

#Step1 引用包
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_digits
from sklearn.tree import DecisionTreeClassifier #决策树模型对象，默认为CART

#Step2 数据加载
digits = load_digits()
data = digits.data

#分割数据，将25%的数据作为测试集，其余作为训练集
train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size = 0.25, random_state = 33)

#Step3 数据预处理
#采用Z-Score规范化
ss = preprocessing.StandardScaler()
train_ss_x = ss.fit_transform(train_x)
test_ss_x = ss.transform(test_x)

#Step4 模型训练
#创建CART分类器
cart = DecisionTreeClassifier()
cart.fit(train_ss_x, train_y)

#Step5 模型评估
predict_y = cart.predict(test_ss_x)
print('CART准确率:%0.4lf' % accuracy_score(test_y, predict_y))
