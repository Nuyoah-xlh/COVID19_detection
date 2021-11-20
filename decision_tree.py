from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report
from data_preprocess import data_preprocess

# 决策树预测


data_pre = data_preprocess()
X_train = data_pre.get_X_train()
train_y = data_pre.get_train_y()
X_test = data_pre.get_X_test()
test_y = data_pre.get_test_y()

# # 建立决策树
Tree = DecisionTreeClassifier(criterion='entropy', max_depth=None)
Tree.fit(X_train, train_y)
print(X_train)
print(X_test)
# 对测试集进行预测
Y_pred2 = Tree.predict(X_test)
print(round(accuracy_score(test_y, Y_pred2), 2) * 100)
print(classification_report(test_y, Y_pred2))
