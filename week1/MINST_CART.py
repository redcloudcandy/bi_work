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
print(data.shape)
print(data[0].reshape(8,8).reshape(64))
#分割数据，将25%的数据作为测试集，其余作为训练集
#$train_x, test_x, train_y, test_y = train_test_split(data, digits.target, test_size = 0.25, random_state = 33)

#Step3 数据预处理
#采用Z-Score规范化
#$ss = preprocessing.StandardScaler()
#$train_ss_x = ss.fit_transform(train_x)
#$test_ss_x = ss.transform(test_x)

#Step4 模型训练
#创建CART分类器
#$cart = DecisionTreeClassifier()
#$cart.fit(train_ss_x, train_y)

#Step5 模型评估
#$predict_y = cart.predict(test_ss_x)
#$print('CART准确率:%0.4lf' % accuracy_score(test_y, predict_y))

'''
iris = datasets.load_iris() #典型分类数据模型
#这里我们数据统一用pandas处理
data = pd.DataFrame(iris.data, columns=iris.feature_names)
data['class'] = iris.target

#这里只取两类
data = data[data['class']!=2]
#为了可视化方便，这里取两个属性为例
X = data[['sepal length (cm)','sepal width (cm)']]
Y = data[['class']]
#划分数据集
X_train, X_test, Y_train, Y_test =train_test_split(X, Y)
#创建决策树模型对象，默认为CART
dt = tree.DecisionTreeClassifier()
dt.fit(X_train, Y_train)

#显示训练结果
print(dt.score(X_test, Y_test)) #score是指分类的正确率

#作图
h = 0.02
x_min, x_max = X.iloc[:, 0].min() - 1, X.iloc[:, 0].max() + 1
y_min, y_max = X.iloc[:, 1].min() - 1, X.iloc[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = dt.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, cmap=plt.cm.Paired)

#做出原来的散点图
class1_x = X.loc[Y['class']==0,'sepal length (cm)']
class1_y = X.loc[Y['class']==0,'sepal width (cm)']
l1 = plt.scatter(class1_x,class1_y,color='b',label=iris.target_names[0])
class1_x = X.loc[Y['class']==1,'sepal length (cm)']
class1_y = X.loc[Y['class']==1,'sepal width (cm)']
l2 = plt.scatter(class1_x,class1_y,color='r',label=iris.target_names[1])
plt.legend(handles = [l1, l2], loc = 'best')

plt.grid(True)
plt.show()

#导出决策树的图片，需要配置graphviz，并且添加到环境变量
dot_data = StringIO()
tree.export_graphviz(dt, out_file=dot_data,feature_names=X.columns,  
                     class_names=['healthy','infected'],
                     filled=True, rounded=True,  
                     special_characters=True)
graph = pydot.graph_from_dot_data(dot_data.getvalue())[0]
graph.write_png("Iris.png")
'''