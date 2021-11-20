# COVID19_detection

## 背景

​	当前，世界正遭受全球COVID19大流行的困扰。数十亿人受到影响，数百万的人员伤亡已经发生。因此，鉴定受SARS-CoV-2病毒感染或已经受其污染的个人至关重要。 这种识别有助于公共卫生组织和政府制定行动计划，以减少这种大流行的影响。从这种意义上讲，Hilab是一家远程实验室公司，它执行数十种类型的血液检查，包括针对COVID19的血清学检查，该公司已经在巴西进行了数百万次检查。为了改善对这种病毒的检测，可以使用机器学习方法来帮助实验室专家进行决策。 因此，本项目将致力于解决构建用于检测COVID19的具有高置信度和准确性的机器学习模型的难题。

## 方法

* 决策树（Decision tree）
* 随机森林（Random forest）
* 支持向量机（SVN）
* 主成分分析（PCA）

## 数据集

数据集地址：https://drive.google.com/drive/folders/1FfIx5WmEc_C7d3Ai7ONIQE4s-o2xQZz5?usp=sharing

## 项目结构

~~~
/
-dataset/		#数据集存放目录
--test/			#测试集目录
---test.csv		#测试集文件
--train/  		#训练集目录
---train_1.csv	#训练集文件1（此文件与测试集相同，默认不使用）
---train_2.csv	#训练集文件2
.......
---train_7.csv	#训练集文件7

-data_preprocess.py	#数据集提取与预处理
-pca.py				#pca降维的相关实验
-decision_tree.py	#决策树
-random_forest.py	#随机森林
-SVM.py				#SVM
-README.md			#说明文件
~~~



