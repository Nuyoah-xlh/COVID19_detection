import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from data_preprocess import data_preprocess

# 绘制降维对信息衰减的相关图像

data_pre = data_preprocess()
X_train = data_pre.get_X_train()
train_y = data_pre.get_train_y()
X_test = data_pre.get_X_test()
test_y = data_pre.get_test_y()

pca = PCA()
# pca=PCA(n_components=0.9)
# X_train = pca.transform(X_train)
pca.fit(X_train, train_y)
ratio = pca.explained_variance_ratio_
print("pca.components_", pca.components_.shape)
print("pca_var_ratio", pca.explained_variance_ratio_.shape)
# 绘制图形
plt.plot([i for i in range(X_train.shape[1] - 12050)],
         [np.sum(ratio[:i + 1]) for i in range(X_train.shape[1] - 12050)])
plt.xticks(np.arange(X_train.shape[1] - 12050, step=10))
plt.yticks(np.arange(0, 1.00, 0.05))
plt.grid()
plt.show()
