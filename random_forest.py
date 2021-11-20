from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import RandomForestClassifier
from data_preprocess import data_preprocess

# 获取数据
data_pre = data_preprocess()
X_train = data_pre.get_X_train()
train_y = data_pre.get_train_y()
X_test = data_pre.get_X_test()
test_y = data_pre.get_test_y()

print("随机森林预测中....")

forest = RandomForestClassifier(n_estimators=110, random_state=234)
forest.fit(X_train, train_y)
# 对测试集进行预测
Y_pred3 = forest.predict(X_test)
print(round(accuracy_score(test_y, Y_pred3), 2) * 100)
print(classification_report(test_y, Y_pred3))
