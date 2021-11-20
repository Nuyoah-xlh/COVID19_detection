from sklearn.svm import SVC
from sklearn.decomposition import PCA

from data_preprocess import data_preprocess

# 获取数据
data_pre = data_preprocess()
X_train = data_pre.get_X_train()
train_y = data_pre.get_train_y()
X_test = data_pre.get_X_test()
test_y = data_pre.get_test_y()

########################PCA降维######################


# 维度 ！！可调！！
n_comp = 7

# while True:

pca = PCA(n_comp)
# finding pca axes
pca.fit(X_train)
# projecting training data onto pca axes
X_train = pca.transform(X_train)
# projecting test data onto pca axes
X_test = pca.transform(X_test)

print(X_train.shape)
print(X_test.shape)

###############参数！可调#################
svm = SVC(gamma=0.001, C=0.01)
svm.fit(X_train, train_y)
# 在验证集上评估SVC
score = svm.score(X_test, test_y)
print("score:")
print(score)

# 找到最合适的参数
# best_score = 0
# best_parameters = 0
# c = [10**(-3), 10**(- 2), 10**(- 1), 10**0, 10**1, 10**2, 10**3]  # possible values of C
# g = [10**(-9), 10**(-7), 10**(-5), 10**(-3)]  # possible values of gamma
# for gamma in g:
#     for C in c:
#         # 对每种参数组合都训练一个SVC
#         svm = SVC(gamma=gamma, C=C)
#         svm.fit(X_train, train_y)
#         # 在验证集上评估SVC
#         score = svm.score(X_test, test_y)
#         print("score:")
#         print(score)
#         # 如果我们得到了更高的分数，则保存该分数和对应的参数
#         if score > best_score:
#             best_score = score
#             best_parameters = {'C': C, 'gamma': gamma}
#
# print("结束")
# # 在训练+验证集上重新构建一个模型，并在测试集上进行评估
# svm = SVC(**best_parameters)
# svm.fit(X_train, train_y)
# test_score = svm.score(X_test, test_y)
# print("Best score on validation set: {:.2f}".format(best_score))
# print("Best parameters: ", best_parameters)
# print("Test set score with best parameters: {:.2f}".format(test_score))
