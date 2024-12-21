#%%load package
from sklearn.ensemble import VotingClassifier
from sklearn import datasets
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import xgboost
from sklearn.ensemble import GradientBoostingClassifier
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm, datasets
from sklearn.metrics import auc
from sklearn.metrics import RocCurveDisplay
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import roc_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn import linear_model
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import label_binarize
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn import tree
from sklearn.neural_network import MLPClassifier
import miceforest as mf
from sklearn.datasets import load_iris
from xgboost.sklearn import XGBClassifier
import xgboost as xgb
from xgboost import plot_importance
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import warnings
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn import metrics
from matplotlib import pyplot
from numpy import argmax
from functools import reduce
from sklearn import metrics
import seaborn as sns
from sklearn import tree
from io import StringIO
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from sklearn.datasets import fetch_openml
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder
from sklearn.metrics import confusion_matrix
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import streamlit as st
import pickle
import sklearn
import json
import random
import matplotlib.pyplot as plt
from xgboost import XGBClassifier
from imblearn.over_sampling import RandomOverSampler
import shap
from collections import Counter
from sklearn.datasets import make_classification
from imblearn.over_sampling import SMOTE
from imblearn.over_sampling import ADASYN
from sklearn.metrics import f1_score
%matplotlib
sns.set()
# =============================================================================
#%% 导入相关的数据
# =============================================================================
new_data = pd.read_csv('E:\\Spyder_2022.3.29\\output\\machinel\\sy_output\\liver_cacer_em\\data.csv')
X_data= new_data[['AFP_400','CEA','CA125','CA199','ALP','TG']]
X_data.info()
#查看数据的空缺值信息
X_data.isnull().sum(axis=0)
# 引入随机森林模型和填补缺失值的模型
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestRegressor
def func(x: str) -> int:
    if x == "First Owner":
        return 1
    elif x == "Second Owner":
        return 2
    elif x == "Third Owner":
        return 3
    
X_data.isnull().sum(axis=0)
#将缺失值的名字纳入
data = X_data
for name in ['CEA','CA125','ALP']:
    X = data.drop(columns=f"{name}")
    Y = data.loc[:, f"{name}"]
    X_0 = SimpleImputer(missing_values=np.nan, strategy="constant").fit_transform(X)
    y_train = Y[Y.notnull()]
    y_test = Y[Y.isnull()]
    x_train = X_0[y_train.index, :]
    x_test = X_0[y_test.index, :]
    rfc = RandomForestRegressor(n_estimators=100)
    rfc = rfc.fit(x_train, y_train)
    y_predict = rfc.predict(x_test)
    data.loc[Y.isnull(), f"{name}"] = y_predict
    
X_data = data 
X_data.isnull().sum(axis=0)
#转换自变量
X = np.array(X_data)
#指定因变量
y = new_data[['M']]
#excel表格中的必须为012，以0最小
y = label_binarize(y, classes=[0,1])
#%%随机数种子
random_state_new = 969
#%%数据不平衡的处理
oversample = SMOTE(random_state=random_state_new)
X, y = oversample.fit_resample(X, y)
counter = Counter(y)
print(counter)
#%%拆分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state_new,test_size=0.3)
#%%建立模型
# =============================================================================
# LR模型
# =============================================================================
logis_model = LogisticRegression(random_state=random_state_new,
                                 solver='lbfgs', multi_class='multinomial')
lr_model = logis_model
lr_model.fit(X_train, y_train)
lr_model.score(X_test, y_test)
# =============================================================================
# K近邻分类模型
# =============================================================================
KNN_model = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
KNN_model.score(X_test, y_test)
# =============================================================================
# 高斯贝叶斯分类模型
# =============================================================================
GaNB_model = GaussianNB().fit(X_train, y_train)
GaNB_model.score(X_test, y_test)
# =============================================================================
# 决策树分类模型
# =============================================================================
tree = tree.DecisionTreeClassifier(random_state=random_state_new)
tree_model = tree.fit(X_train, y_train)
tree_model.score(X_test, y_test)
# =============================================================================
# Bagging分类模型
# =============================================================================
Bag = BaggingClassifier(KNeighborsClassifier(),
                        max_samples=0.5, max_features=0.5, random_state=random_state_new)
Bag_model = Bag.fit(X_train, y_train)
Bag_model.score(X_test, y_test)
# =============================================================================
# 随机森林模型
# =============================================================================
RF = RandomForestClassifier(n_estimators=10, max_depth=3,
                            min_samples_split=12, random_state=random_state_new)
RF_model = RF.fit(X_train, y_train)
RF_model.score(X_test, y_test)
# =============================================================================
# 极端随机树分类模型
# =============================================================================
ET = ExtraTreesClassifier(n_estimators=10, max_depth=None,
                          min_samples_split=2, random_state=random_state_new)
ET_model = ET.fit(X_train, y_train)
ET_model.score(X_test, y_test)
# =============================================================================
# AdaBoost模型
# =============================================================================
AB = AdaBoostClassifier(n_estimators=10, random_state=random_state_new)
AB_model = AB.fit(X_train, y_train)
AB_model.score(X_test, y_test)
# =============================================================================
# GBDT模型
# =============================================================================
GBT = GradientBoostingClassifier(
    n_estimators=10, learning_rate=1.0, max_depth=1, random_state=random_state_new)
GBT_model = GBT.fit(X_train, y_train)
GBT_model.score(X_test, y_test)
# =============================================================================
# VOTE模型
# =============================================================================
clf1 = LogisticRegression(
    solver='lbfgs', multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=random_state_new)
clf3 = GaussianNB()
VOTE = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
VOTE_model = VOTE.fit(X_train, y_train)
VOTE_model.score(X_test, y_test)
# =============================================================================
# GBM模型
# =============================================================================
gbm = GradientBoostingClassifier(
    n_estimators=100, learning_rate=1, max_depth=1, random_state=random_state_new)
gbm_model = gbm.fit(X_train, y_train)
gbm_model.score(X_test, y_test)
# =============================================================================
# XGboost模型
# =============================================================================
xgb_model = xgb.XGBClassifier(
    n_estimators=100, max_depth=2, learning_rate=1.85, random_state=random_state_new)
xgb_model = xgb_model.fit(X_train, y_train)
xgb_model.score(X_test, y_test)
# =============================================================================
# MLP模型
# =============================================================================
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='lbfgs',
                    alpha=0.0001,
                    batch_size='auto',
                    learning_rate='constant',
                    learning_rate_init=1,
                    power_t=0.5,
                    max_iter=200,
                    shuffle=True, random_state=random_state_new)
mlp_model = mlp_model.fit(X_train, y_train)
lr_model = logis_model
#%%交叉验证LR、AB、BAG、MLP、GBM、XGB
from sklearn.model_selection import cross_val_score,StratifiedKFold,LeaveOneOut
strKFold = StratifiedKFold(n_splits=10,shuffle=True,random_state=random_state_new)
cv=strKFold
result_lr=cross_val_score(lr_model,X_train,y_train,scoring='roc_auc',cv=cv,n_jobs=-1)
result_ab=cross_val_score(AB_model,X_train,y_train,scoring='roc_auc',cv=cv,n_jobs=-1)
result_bag=cross_val_score(Bag_model,X_train,y_train,scoring='roc_auc',cv=cv,n_jobs=-1)
result_mlp=cross_val_score(mlp_model,X_train,y_train,scoring='roc_auc',cv=cv,n_jobs=-1)
result_gbm=cross_val_score(gbm_model,X_train,y_train,scoring='roc_auc',cv=cv,n_jobs=-1)
result_xgb=cross_val_score(xgb_model,X_train,y_train,scoring='roc_auc',cv=cv,n_jobs=-1)
#%%折线图的绘制LR、AB、BAG、MLP、GBM、XGB
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
plt.plot(x, result_lr, label='LR: Average AUC = {:.3f}, STD = {:.3f}'.format(result_lr.mean(), result_lr.std()),
         linewidth=3, color='#fe5722', marker='>', markerfacecolor='#fe5722', markersize=12)
plt.plot(x, result_ab, label='AB: Average AUC = {:.3f}, STD = {:.3f}'.format(result_ab.mean(), result_ab.std()),
         linewidth=3, color='#03a8f3', marker='>', markerfacecolor='#03a8f3', markersize=12)
plt.plot(x, result_mlp, label='MLP: Average AUC = {:.3f}, STD = {:.3f}'.format(result_mlp.mean(), result_mlp.std()),
         linewidth=3, color='#009587', marker='>', markerfacecolor='#009587', markersize=12)
plt.plot(x, result_bag, label='BAG: Average AUC = {:.3f}, STD = {:.3f}'.format(result_bag.mean(), result_bag.std()),
         linewidth=3, color='#673ab6', marker='>', markerfacecolor='#673ab6', markersize=12)
plt.plot(x, result_gbm, label='GBM: Average AUC = {:.3f}, STD = {:.3f}'.format(result_gbm.mean(), result_gbm.std()),
         linewidth=3, color='#b5da3d', marker='>', markerfacecolor='#b5da3d', markersize=12)
plt.plot(x, result_xgb, label='XGB: Average AUC = {:.3f}, STD = {:.3f}'.format(result_xgb.mean(), result_xgb.std()),
         linewidth=3, color='#3f51b4', marker='>', markerfacecolor='#3f51b4', markersize=12)
x_major_locator = MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
plt.ylim(0.8, 1.02)
plt.xlim(0.7, 10.3)
plt.xlabel('Round of Cross')
plt.ylabel('AUC')
plt.title('Ten Fold Cross Validation')
plt.legend(loc=4)
plt.show()
#%%train and test ROC曲线LR、AB、MLP、BAG、GBM、XGB
plt.style.use('tableau-colorblind10')
def plot_roc(k,y_pred_undersample_score,labels_test,classifiers,color,title):
    fpr, tpr, thresholds = metrics.roc_curve(labels_test.values.ravel(),y_pred_undersample_score)
    roc_auc = metrics.auc(fpr,tpr)
    plt.figure(figsize=(20,16))
    plt.figure(k)
    plt.title(title)
    plt.plot(fpr, tpr, 'b',color=color,label='%s AUC = %0.3f'% (classifiers,roc_auc))
    plt.legend(loc='lower right')
    plt.plot([0,1],[0,1],'r--')
    plt.xlim([-0.02,1.02])
    plt.ylim([-0.02,1.02])
    plt.ylabel('Sensitivity')
    plt.xlabel('1 - Specifity')
fig = plt.gcf()
plt.subplot(1,2,1)
plot_roc(1,lr_model.predict_proba(X_train)[:,1],pd.DataFrame(y_train),'LR','#fe5722','ROC curve of train set')
plot_roc(1,AB_model.predict_proba(X_train)[:,1],pd.DataFrame(y_train),'AB','#03a8f3','ROC curve of train set')
plot_roc(1,mlp_model.predict_proba(X_train)[:,1],pd.DataFrame(y_train),'MLP','#009587','ROC curve of train set')
plot_roc(1,Bag_model.predict_proba(X_train)[:,1],pd.DataFrame(y_train),'BAG','#673ab6','ROC curve of train set')
plot_roc(1,gbm_model.predict_proba(X_train)[:,1],pd.DataFrame(y_train),'GBM','#b5da3d','ROC curve of train set')
plot_roc(1,xgb_model.predict_proba(X_train)[:,1],pd.DataFrame(y_train),'XGB','#3f51b4','ROC curve of train set')
plt.subplot(1,2,2)
plot_roc(1,lr_model.predict_proba(X_test)[:,1],pd.DataFrame(y_test),'LR','#fe5722','ROC curve of test set')
plot_roc(1,AB_model.predict_proba(X_test)[:,1],pd.DataFrame(y_test),'AB','#03a8f3','ROC curve of test set')
plot_roc(1,mlp_model.predict_proba(X_test)[:,1],pd.DataFrame(y_test),'MLP','#009587','ROC curve of test set')
plot_roc(1,Bag_model.predict_proba(X_test)[:,1],pd.DataFrame(y_test),'BAG','#673ab6','ROC curve of test set')
plot_roc(1,gbm_model.predict_proba(X_test)[:,1],pd.DataFrame(y_test),'GBM','#b5da3d','ROC curve of test set')
plot_roc(1,xgb_model.predict_proba(X_test)[:,1],pd.DataFrame(y_test),'XGB','#3f51b4','ROC curve of test set')
plt.show()
#%% PR曲线
from sklearn.metrics import precision_recall_curve, average_precision_score
def ro_curve(k,y_pred, y_label, method_name,color,title):
    y_label = np.array(y_label)
    y_pred = np.array(y_pred)    
    lr_precision, lr_recall, _ = precision_recall_curve(y_label, y_pred)    
    plt.figure(k)
    plt.plot(lr_recall, lr_precision, lw = 2, label= method_name + ' (Area = %0.3f)' % average_precision_score(y_label, y_pred),color=color)
    fontsize = 14
    plt.xlabel('Recall', fontsize = fontsize)
    plt.ylabel('Precision', fontsize = fontsize)
    plt.title(title)
    plt.legend(loc='lower right')
fig = plt.gcf()
#train 
plt.subplot(1,2,1)
ro_curve(1,lr_model.predict_proba(X_train)[:,1],y_train,'LR','red','Precision Recall Curve of train set')
ro_curve(1,AB_model.predict_proba(X_train)[:,1],y_train,'AB','blue','Precision Recall Curve of train set')
ro_curve(1,mlp_model.predict_proba(X_train)[:,1],y_train,'MLP','green','Precision Recall Curve of train set')
ro_curve(1,Bag_model.predict_proba(X_train)[:,1],y_train,'BAG','m','Precision Recall Curve of train set')
ro_curve(1,RF_model.predict_proba(X_train)[:,1],y_train,'GBM','tomato','Precision Recall Curve of train set')
ro_curve(1,gbm_model.predict_proba(X_train)[:,1],y_train,'XGB','darkblue','Precision Recall Curve of train set')
#test
plt.subplot(1,2,2)
ro_curve(1,lr_model.predict_proba(X_test)[:,1],y_test,'LR','red','Precision Recall Curve of test set')
ro_curve(1,AB_model.predict_proba(X_test)[:,1],y_test,'AB','blue','Precision Recall Curve of test set')
ro_curve(1,mlp_model.predict_proba(X_test)[:,1],y_test,'MLP','green','Precision Recall Curve of test set')
ro_curve(1,Bag_model.predict_proba(X_test)[:,1],y_test,'BAG','m','Precision Recall Curve of test set')
ro_curve(1,RF_model.predict_proba(X_test)[:,1],y_test,'GBM','tomato','Precision Recall Curve of test set')
ro_curve(1,gbm_model.predict_proba(X_test)[:,1],y_test,'XGB','darkblue','Precision Recall Curve of test set')
plt.show()
#%% calibration_curve
rf_prob = AB_model.predict(X_test)
lr_prob = logis_model.predict(X_test)
dt_prob = Bag_model.predict(X_test)
mlp_prob = mlp_model.predict(X_test)
gbm_prob = gbm_model.predict(X_test)
xgb_prob = xgb_model.predict(X_test)
plt.rcParams["axes.grid"] = False
sns.set()
from sklearn.calibration import calibration_curve
def calibration_curve_1(k,y_pred,y_true,method_name,color,title):
    prob_true, prob_pred = calibration_curve(y_true, y_pred, n_bins=5)
    plt.figure(k)
    plt.plot(prob_pred,prob_true,color=color,label='%s calibration_curve'%method_name,marker='s')
    plt.plot([i/100 for i in range(0,100)],[i/100 for i in range(0,100)],color='black',linestyle='--')
    plt.xlim(-0.02,1.02,0.2)
    plt.ylim(-0.02,1.02,0.2)
    plt.xlabel('y_preds')
    plt.ylabel('y_real')
    plt.title(title)
    plt.legend(loc='lower right')
plt.subplot(1,2,1)
calibration_curve_1(1,logis_model.predict_proba(X_train)[:,1],y_train,'LR','red','Calibration curve of train set')
calibration_curve_1(1,AB_model.predict_proba(X_train)[:,1],y_train,'AB','blue','Calibration curve of train set')
calibration_curve_1(1,RF_model.predict_proba(X_train)[:,1],y_train,'MLP','green','Calibration curve of train set')
calibration_curve_1(1,Bag_model.predict_proba(X_train)[:,1],y_train,'BAG','m','Calibration curve of train set')
calibration_curve_1(1,gbm_model.predict_proba(X_train)[:,1],y_train,'GBM','tomato','Calibration curve of train set')
calibration_curve_1(1,mlp_model.predict_proba(X_train)[:,1],y_train,'XGB','darkblue','Calibration curve of train set')
plt.subplot(1,2,2)
calibration_curve_1(1,logis_model.predict_proba(X_test)[:,1],y_test,'LR','red','Calibration curve of test set')
calibration_curve_1(1,AB_model.predict_proba(X_test)[:,1],y_test,'AB','blue','Calibration curve of test set')
calibration_curve_1(1,RF_model.predict_proba(X_test)[:,1],y_test,'MLP','green','Calibration curve of test set')
calibration_curve_1(1,Bag_model.predict_proba(X_test)[:,1],y_test,'BAG','m','Calibration curve of test set')
calibration_curve_1(1,gbm_model.predict_proba(X_test)[:,1],y_test,'GBM','tomato','Calibration curve of test set')
calibration_curve_1(1,mlp_model.predict_proba(X_test)[:,1],y_test,'XGB','darkblue','Calibration curve of test set')
plt.show()
#%%建立模型
jc = 10
lr = LogisticRegression(penalty="none", random_state=random_state_new)
rf = RandomForestClassifier(
    n_estimators=200,  max_features=4, random_state=random_state_new)
dt = tree.DecisionTreeClassifier(
    min_weight_fraction_leaf=0.25, random_state=random_state_new)
mlp = MLPClassifier(hidden_layer_sizes=(100,), activation='relu', solver='lbfgs',
                    alpha=0.0001,
                    batch_size='auto',
                    learning_rate='constant',
                    learning_rate_init=0.01,
                    power_t=0.5,
                    max_iter=200,
                    shuffle=True, random_state=random_state_new)
xgb_model = xgb.XGBClassifier(
    n_estimators=360, max_depth=2, learning_rate=1, random_state=random_state_new)
gbm = GradientBoostingClassifier(
    n_estimators=100, learning_rate=1, max_depth=1, random_state=random_state_new)
# =============================================================================
# 划分数据集
# =============================================================================
# Split the data into a training set and a test set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=random_state_new)
# =============================================================================
# 分类器模型的导入
# =============================================================================
# =============================================================================
# 逻辑回归模型
# =============================================================================
logis_model = LogisticRegression(random_state=random_state_new,
                                 solver='lbfgs', multi_class='multinomial').fit(X_train, y_train)
logis_model.score(X_test, y_test)
# =============================================================================
# K近邻分类模型
# =============================================================================
KNN_model = KNeighborsClassifier(n_neighbors=3).fit(X_train, y_train)
KNN_model.score(X_test, y_test)
# =============================================================================
# 高斯贝叶斯分类模型
# =============================================================================
GaNB_model = GaussianNB().fit(X_train, y_train)
GaNB_model.score(X_test, y_test)
# =============================================================================
# 决策树分类模型
# =============================================================================
tree = tree.DecisionTreeClassifier(random_state=random_state_new)
tree_model = tree.fit(X_train, y_train)
tree_model.score(X_test, y_test)
# =============================================================================
# Bagging分类模型
# =============================================================================
Bag = BaggingClassifier(KNeighborsClassifier(
), max_samples=0.5, max_features=0.5, random_state=random_state_new)
Bag_model = Bag.fit(X_train, y_train)
Bag_model.score(X_test, y_test)
# =============================================================================
# 随机森林模型
# =============================================================================
RF = RandomForestClassifier(n_estimators=10, max_depth=3,
                            min_samples_split=12, random_state=random_state_new)
RF_model = RF.fit(X_train, y_train)
RF_model.score(X_test, y_test)
# =============================================================================
# 极端随机树分类模型
# =============================================================================
ET = ExtraTreesClassifier(n_estimators=10, max_depth=None,
                          min_samples_split=2, random_state=random_state_new)
ET_model = ET.fit(X_train, y_train)
ET_model.score(X_test, y_test)
# =============================================================================
# AdaBoost模型
# =============================================================================
AB = AdaBoostClassifier(n_estimators=10, random_state=random_state_new)
AB_model = AB.fit(X_train, y_train)
AB_model.score(X_test, y_test)
# =============================================================================
# GBDT模型
# =============================================================================
GBT = GradientBoostingClassifier(
    n_estimators=10, learning_rate=1.0, max_depth=1, random_state=random_state_new)
GBT_model = GBT.fit(X_train, y_train)
GBT_model.score(X_test, y_test)
# =============================================================================
# VOTE模型
# =============================================================================
clf1 = LogisticRegression(
    solver='lbfgs', multi_class='multinomial', random_state=1)
clf2 = RandomForestClassifier(n_estimators=50, random_state=random_state_new)
clf3 = GaussianNB()
VOTE = VotingClassifier(
    estimators=[('lr', clf1), ('rf', clf2), ('gnb', clf3)], voting='hard')
VOTE_model = VOTE.fit(X_train, y_train)
VOTE_model.score(X_test, y_test)
# =============================================================================
# GBM模型
# =============================================================================
gbm = GradientBoostingClassifier(
    n_estimators=100, learning_rate=1, max_depth=1, random_state=random_state_new)
gbm_model = gbm.fit(X_train, y_train)
gbm_model.score(X_test, y_test)
# =============================================================================
# XGboost模型
# =============================================================================
xgb_model = xgb.XGBClassifier(
    n_estimators=360, max_depth=2, learning_rate=1, random_state=random_state_new)
xgb_model = xgb_model.fit(X_train, y_train)
xgb_model.score(X_test, y_test)
# =============================================================================
# 模型准确率的比较
# =============================================================================
print("Logistic回归的模型准确率：{:.3f}".format(logis_model.score(X_test, y_test)))
print("KNN回归的模型准确率：{:.3f}".format(KNN_model.score(X_test, y_test)))
print("高斯贝叶分类器的模型准确率：{:.3f}".format(GaNB_model.score(X_test, y_test)))
print("决策树分类器的模型准确率：{:.3f}".format(tree_model.score(X_test, y_test)))
print("Bagging分类模型的模型准确率：{:.3f}".format(Bag_model.score(X_test, y_test)))
print("随机森林分类模型的模型准确率：{:.3f}".format(RF_model.score(X_test, y_test)))
print("极端随机树分类模型的模型准确率：{:.3f}".format(ET_model.score(X_test, y_test)))
print("AdaBoost模型的模型准确率：{:.3f}".format(AB_model.score(X_test, y_test)))
print("GBDT模型的模型准确率：{:.3f}".format(GBT_model.score(X_test, y_test)))
print("VOTE模型的模型准确率：{:.3f}".format(VOTE_model.score(X_test, y_test)))
print("GBM模型的模型准确率：{:.3f}".format(gbm_model.score(X_test, y_test)))
print("XGboost模型的模型准确率：{:.3f}".format(xgb_model.score(X_test, y_test)))
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# =============================================================================
# # logis_model
# =============================================================================
cv = StratifiedKFold(n_splits=jc)
tprs_logis_model = []
aucs_logis_model = []
mean_fpr_logis_model = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
# 采用逻辑回归的交叉验证方法
for i, (train, test) in enumerate(cv.split(X, y)):
    logis_model.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        logis_model,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr_logis_model, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs_logis_model.append(interp_tpr)
    aucs_logis_model.append(viz.roc_auc)
aucs_logis_model
# # =============================================================================
# # 采用随机森林的交叉验证
# # =============================================================================
cv = StratifiedKFold(n_splits=jc)
tprs_RF_model = []
aucs_RF_model = []
mean_fpr_RF_model = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
# 采用逻辑回归的交叉验证方法
for i, (train, test) in enumerate(cv.split(X, y)):
    RF_model.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        RF_model,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr_RF_model, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs_RF_model.append(interp_tpr)
    aucs_RF_model.append(viz.roc_auc)
aucs_RF_model
# # =============================================================================
# # 采用决策树的交叉验证
# # =============================================================================
cv = StratifiedKFold(n_splits=10)
tprs_tree_model = []
aucs_tree_model = []
mean_fpr_tree_model = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
# 采用逻辑回归的交叉验证方法
for i, (train, test) in enumerate(cv.split(X, y)):
    tree_model.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        tree_model,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr_tree_model, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs_tree_model.append(interp_tpr)
    aucs_tree_model.append(viz.roc_auc)
aucs_tree_model
# # =============================================================================
# # gbm的交叉验证
# # =============================================================================
cv = StratifiedKFold(n_splits=jc)
tprs_gbm_model = []
aucs_gbm_model = []
mean_fpr_gbm_model = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
# 采用逻辑回归的交叉验证方法
for i, (train, test) in enumerate(cv.split(X, y)):
    gbm_model.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        gbm_model,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr_gbm_model, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs_gbm_model.append(interp_tpr)
    aucs_gbm_model.append(viz.roc_auc)
aucs_gbm_model
# # # =============================================================================
# # # 采用多层感知机的交叉验证
# # # =============================================================================
cv = StratifiedKFold(n_splits=jc)
tprs_mlp_model = []
aucs_mlp_model = []
mean_fpr_mlp_model = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
# 采用逻辑回归的交叉验证方法
for i, (train, test) in enumerate(cv.split(X, y)):
    mlp.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        mlp,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr_mlp_model, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs_mlp_model.append(interp_tpr)
    aucs_mlp_model.append(viz.roc_auc)
aucs_mlp_model
# # =============================================================================
# # 采用xgb的交叉验证
# # =============================================================================
cv = StratifiedKFold(n_splits=jc)
tprs_xgb_model = []
aucs_xgb_model = []
mean_fpr_xgb_model = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
# 采用逻辑回归的交叉验证方法
for i, (train, test) in enumerate(cv.split(X, y)):
    xgb_model.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        xgb_model,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr_xgb_model, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs_xgb_model.append(interp_tpr)
    aucs_xgb_model.append(viz.roc_auc)
aucs_xgb_model
# # =============================================================================
# # 采用AB_model的交叉验证
# # =============================================================================
cv = StratifiedKFold(n_splits=jc)
tprs_AB_model = []
aucs_AB_model = []
mean_fpr_AB_model = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
# 采用逻辑回归的交叉验证方法
for i, (train, test) in enumerate(cv.split(X, y)):
    AB_model.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        AB_model,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr_AB_model, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs_AB_model.append(interp_tpr)
    aucs_AB_model.append(viz.roc_auc)
aucs_AB_model
# # =============================================================================
# # 采用Bagging分类模型的交叉验证
# # =============================================================================
cv = StratifiedKFold(n_splits=jc)
tprs_Bag_model = []
aucs_Bag_model = []
mean_fpr_Bag_model = np.linspace(0, 1, 100)
fig, ax = plt.subplots()
# 采用逻辑回归的交叉验证方法
for i, (train, test) in enumerate(cv.split(X, y)):
    Bag_model.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        Bag_model,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr_Bag_model, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs_Bag_model.append(interp_tpr)
    aucs_Bag_model.append(viz.roc_auc)
aucs_Bag_model
# # =============================================================================
# # 画ROC曲线
# # =============================================================================
# # sns.set()
# # fig, ax = plt.subplots()
# # ax.plot([0, 1], [0, 1], linestyle="--", lw=2,
# #         color="r", label="Chance", alpha=0.8)
# # mean_tpr_lr = np.mean(tprs_lr, axis=0)
# # mean_tpr_clf = np.mean(tprs_clf, axis=0)
# # mean_tpr_rf = np.mean(tprs_rf, axis=0)
# # mean_tpr_mlp = np.mean(tprs_mlp, axis=0)
# # mean_tpr_xgb_model = np.mean(tprs_xgb_model, axis=0)
# # mean_tpr_lr[-1] = 1.0
# # mean_tpr_clf[-1] = 1.0
# # mean_tpr_rf[-1] = 1.0
# # mean_tpr_mlp[-1] = 1.0
# # mean_tpr_xgb_model[-1] = 1.0
# # mean_auc_lr = auc(mean_fpr_lr, mean_tpr_lr)
# # std_auc = np.std(aucs)
# # ax.plot(
# #     mean_fpr,
# #     mean_tpr,
# #     color="b",
# #     label=r"Mean ROC (AUC = %0.2f $\pm$ %0.2f)" % (mean_auc, std_auc),
# #     lw=2,
# #     alpha=0.8,
# # )
# # std_tpr = np.std(tprs, axis=0)
# # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# # ax.fill_between(
# #     mean_fpr,
# #     tprs_lower,
# #     tprs_upper,
# #     color="grey",
# #     alpha=0.2,
# #     label=r"$\pm$ 1 std. dev.",
# # )
# # ax.set(
# #     xlim=[-0.05, 1.05],
# #     ylim=[-0.05, 1.05],
# #     title="Receiver operating characteristic example",
# # )
# # ax.legend(loc="lower right")
# # plt.show()
# # =============================================================================
# # 选择模型
# # =============================================================================
# # print("Logistic回归的模型准确率：{:.3f}".format(logis_model.score(X_test, y_test)))
# # print("KNN回归的模型准确率：{:.3f}".format(KNN_model.score(X_test, y_test)))
# # print("高斯贝叶分类器的模型准确率：{:.3f}".format(GaNB_model.score(X_test, y_test)))
# # print("决策树分类器的模型准确率：{:.3f}".format(tree_model.score(X_test, y_test)))
# # print("Bagging分类模型的模型准确率：{:.3f}".format(Bag_model.score(X_test, y_test)))
# # print("随机森林分类模型的模型准确率：{:.3f}".format(RF_model.score(X_test, y_test)))
# # print("极端随机树分类模型的模型准确率：{:.3f}".format(ET_model.score(X_test, y_test)))
# # print("AdaBoost模型的模型准确率：{:.3f}".format(AB_model.score(X_test, y_test)))
# # print("GBDT模型的模型准确率：{:.3f}".format(GBT_model.score(X_test, y_test)))
# # print("VOTE模型的模型准确率：{:.3f}".format(VOTE_model.score(X_test, y_test)))
# # print("GBM模型的模型准确率：{:.3f}".format(gbm_model.score(X_test, y_test)))
# # print("XGboost模型的模型准确率：{:.3f}".format(xgb_model.score(X_test, y_test)))
# # =============================================================================
# # 折线图绘制
# # =============================================================================
# # 注意GBM和LR的倒用
x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# x = [1, 2, 3, 4, 5]
fig, ax = plt.subplots(figsize=(12, 8), dpi=100)
# plt.plot(x,aucs_clf,label='CLF: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_clf).mean(), np.array(aucs_clf).std()),
#           linewidth=3,color='#03a8f3',marker='$\heartsuit$', markerfacecolor='#03a8f3',markersize=12)
# plt.plot(x,aucs_lr,label='LR: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_lr).mean(), np.array(aucs_lr).std()),
#           linewidth=3,color='#fe5722',marker='$\heartsuit$', markerfacecolor='#fe5722',markersize=12)
# plt.plot(x,aucs_mlp,label='MLP: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_mlp).mean(), np.array(aucs_mlp).std()),
#           linewidth=3,color='#009587',marker='$\heartsuit$', markerfacecolor='#009587',markersize=12)
# plt.plot(x,aucs_rf,label='RF: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_rf).mean(), np.array(aucs_rf).std()),
#           linewidth=3,color='#673ab6',marker='$\heartsuit$', markerfacecolor='#673ab6',markersize=12)
# plt.plot(x,aucs_xgb_model,label='XGB: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_xgb_model).mean(), np.array(aucs_xgb_model).std()),
#           linewidth=3,color='#3f51b4',marker='$\heartsuit$', markerfacecolor='#3f51b4',markersize=12)
plt.plot(x, aucs_AB_model, label='AB: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_AB_model).mean(), np.array(aucs_AB_model).std()),
          linewidth=3, color='#03a8f3', marker='>', markerfacecolor='#03a8f3', markersize=12)
plt.plot(x, aucs_logis_model, label='LR: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_logis_model).mean(), np.array(aucs_logis_model).std()),
          linewidth=3, color='#f59f00', marker='>', markerfacecolor='#f59f00', markersize=12)
plt.plot(x, aucs_mlp_model, label='GBM: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_mlp_model).mean(), np.array(aucs_mlp_model).std()),
          linewidth=3, color='#c2255c', marker='>', markerfacecolor='#c2255c', markersize=12)
plt.plot(x, aucs_Bag_model, label='BAG: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_Bag_model).mean(), np.array(aucs_Bag_model).std()),
          linewidth=3, color='#9775fa', marker='>', markerfacecolor='#9775fa', markersize=12)
plt.plot(x, aucs_gbm_model, label='GBM: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_gbm_model).mean(), np.array(aucs_gbm_model).std()),
          linewidth=3, color='#b5da3d', marker='>', markerfacecolor='#b5da3d', markersize=12)
plt.plot(x, aucs_xgb_model, label='XGB: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_xgb_model).mean(), np.array(aucs_xgb_model).std()),
          linewidth=3, color='#38d9a9', marker='>', markerfacecolor='#38d9a9', markersize=12)
x_major_locator = MultipleLocator(1)
# 把x轴的刻度间隔设置为1，并存在变量里
# y_major_locator=MultipleLocator(10)
# 把y轴的刻度间隔设置为10，并存在变量里
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
# 把x轴的主刻度设置为1的倍数
# ax.yaxis.set_major_locator(y_major_locator)
# 把y轴的主刻度设置为10的倍数
# plt.xlim(-0.5,11)
# #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
# plt.ylim(-5,110)
# #把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
# plt.show()
plt.ylim(0.7, 1.01)
plt.xlim(0.7, 10.3)
# plt.xlim(0.7, 5.3)
plt.xlabel('Round of Cross')
plt.ylabel('AUC')
plt.title('Ten Fold Cross Validation')
# plt.title('Five Fold Cross Validation')
plt.legend(loc=4)
plt.show()
# print("Logistic回归的模型准确率：{:.3f}".format(logis_model.score(X_test, y_test)))
# print("KNN回归的模型准确率：{:.3f}".format(KNN_model.score(X_test, y_test)))
# print("高斯贝叶分类器的模型准确率：{:.3f}".format(GaNB_model.score(X_test, y_test)))
# print("决策树分类器的模型准确率：{:.3f}".format(tree_model.score(X_test, y_test)))
# print("Bagging分类模型的模型准确率：{:.3f}".format(Bag_model.score(X_test, y_test)))
# print("随机森林分类模型的模型准确率：{:.3f}".format(RF_model.score(X_test, y_test)))
# print("极端随机树分类模型的模型准确率：{:.3f}".format(ET_model.score(X_test, y_test)))
# print("AdaBoost模型的模型准确率：{:.3f}".format(AB_model.score(X_test, y_test)))
# print("GBDT模型的模型准确率：{:.3f}".format(GBT_model.score(X_test, y_test)))
# print("VOTE模型的模型准确率：{:.3f}".format(VOTE_model.score(X_test, y_test)))
# print("GBM模型的模型准确率：{:.3f}".format(gbm_model.score(X_test, y_test)))
# print("XGboost模型的模型准确率：{:.3f}".format(xgb_model.score(X_test, y_test)))
# plt.savefig('E:\\Spyder_2022.3.29\\output\\machinel\\lrq_output\lrq_pre_r\\Ten Fold Cross Validation.pdf')
# plt.savefig('E:\\Spyder_2022.3.29\\output\\machinel\\lrq_output\\liver_cutt_r_pre\\Five Fold Cross Validation.pdf')
# # # # # =============================================================================
# # # # # 另外一种markers
# # # # # =============================================================================
# plt.plot(x,aucs_clf,label='CLF: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_clf).mean(), np.array(aucs_clf).std()),
#           linewidth=3,color='#03a8f3',marker='$\heartsuit$', markerfacecolor='#03a8f3',markersize=12)
# plt.plot(x,aucs_lr,label='LR: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_lr).mean(), np.array(aucs_lr).std()),
#           linewidth=3,color='#fe5722',marker='$\heartsuit$', markerfacecolor='#fe5722',markersize=12)
# plt.plot(x,aucs_mlp,label='MLP: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_mlp).mean(), np.array(aucs_mlp).std()),
#           linewidth=3,color='#009587',marker='$\heartsuit$', markerfacecolor='#009587',markersize=12)
# plt.plot(x,aucs_rf,label='RF: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_rf).mean(), np.array(aucs_rf).std()),
#           linewidth=3,color='#673ab6',marker='$\heartsuit$', markerfacecolor='#673ab6',markersize=12)
# plt.plot(x,aucs_xgb_model,label='XGB: Average AUC = {:.3f}, STD = {:.3f}'.format(np.array(aucs_xgb_model).mean(), np.array(aucs_xgb_model).std()),
#           linewidth=3,color='#3f51b4',marker='$\heartsuit$', markerfacecolor='#3f51b4',markersize=12)
# ax=plt.gca()
# # ax为两条坐标轴的实例
# x_major_locator = MultipleLocator(1)
# # 把x轴的刻度间隔设置为1，并存在变量里
# # y_major_locator=MultipleLocator(10)
# # 把y轴的刻度间隔设置为10，并存在变量里
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# # 把x轴的主刻度设置为1的倍数
# # ax.yaxis.set_major_locator(y_major_locator)
# # 把y轴的主刻度设置为10的倍数
# # plt.xlim(-0.5,11)
# # #把x轴的刻度范围设置为-0.5到11，因为0.5不满一个刻度间隔，所以数字不会显示出来，但是能看到一点空白
# # plt.ylim(-5,110)
# # #把y轴的刻度范围设置为-5到110，同理，-5不会标出来，但是能看到一点空白
# # plt.show()
# plt.ylim(0.3, 1)
# plt.xlim(0.7, 10.3)
# plt.xlabel('Round of Cross')
# plt.ylabel('AUC')
# plt.title('Ten Fold Cross Validation')
# plt.legend(loc=4)
# plt.show()
# plt.savefig('E:\\Spyder_2022.3.29\\output\\machinel\\Ten Fold Cross Validation.pdf')
# # # # ============= 变量重要性可视化 =============
# # # # =============================================================================
# # # feature_names = [["BQL", "SBSL", "HHD", "FHH", "diabetes", "memory", "HSD", "PS", "BMI",
# # #                   "waistline", "NC", "SmA", "SnT", "SuT", "ESS", "AHI",
# # #                   "OAL", "ageper10", "minPper10", "P90per10"]]
# # # str(feature_names)
# # # =============================================================================
# # # 某个机器学习算法的
# # # =============================================================================
# # # lr 5折交叉ROC
# # # =============================================================================
# sns.set()
# cv = StratifiedKFold(n_splits=5)
# tprs = []
# aucs = []
# mean_fpr = np.linspace(0, 1, 100)
# # fig = plt.figure(figsize=(2, 2), dpi=300)
# fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
# for i, (train, test) in enumerate(cv.split(X, y)):
#     logis_model.fit(X[train], y[train])
#     viz = RocCurveDisplay.from_estimator(
#         logis_model,
#         X[test],
#         y[test],
#         name="ROC fold {}".format(i),
#         alpha=0.3,
#         lw=1,
#         ax=ax,
#     )
#     interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
#     interp_tpr[0] = 0.0
#     tprs.append(interp_tpr)
#     aucs.append(viz.roc_auc)
# ax.plot([0, 1], [0, 1], linestyle="--", lw=2,
#         color="r", label="Reference", alpha=0.8)
# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# std_auc = np.std(aucs)
# ax.plot(
#     mean_fpr,
#     mean_tpr,
#     color="b",
#     label=r"Mean ROC (AUC = %0.2f $\pm$ %0.3f)" % (mean_auc, std_auc),
#     lw=2,
#     alpha=0.8,
# )
# std_tpr = np.std(tprs, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# ax.fill_between(
#     mean_fpr,
#     tprs_lower,
#     tprs_upper,
#     color="grey",
#     alpha=0.2,
#     label=r"$\pm$ 1 std. dev.",
# )
# ax.set(
#     xlim=[-0.05, 1.05],
#     ylim=[-0.05, 1.05],
#     title="LR Receiver operating characteristic ",
# )
# ax.legend(loc="lower right")
# plt.xlabel("1 - Specifity")
# plt.ylabel("Sensitivity")
# plt.show()
# plt.savefig('E:\\Spyder_2022.3.29\\output\\machinel\\lrq_output\lrq_pre_r\\LR Receiver operating characteristic.pdf')
# # # # # =============================================================================
# # # # # =============================================================================
# # # # # 决策树机器学习算法
# # # # # =============================================================================
# cv = StratifiedKFold(n_splits=5)
# tprs = []
# aucs = []
# mean_fpr = np.linspace(0, 1, 100)
# # fig = plt.figure(figsize=(2, 2), dpi=300)
# fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
# for i, (train, test) in enumerate(cv.split(X, y)):
#     AB_model.fit(X[train], y[train])
#     viz = RocCurveDisplay.from_estimator(
#         AB_model,
#         X[test],
#         y[test],
#         name="ROC fold {}".format(i),
#         alpha=0.3,
#         lw=1,
#         ax=ax,
#     )
#     interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
#     interp_tpr[0] = 0.0
#     tprs.append(interp_tpr)
#     aucs.append(viz.roc_auc)
# ax.plot([0, 1], [0, 1], linestyle="--", lw=2,
#         color="r", label="Reference", alpha=0.8)
# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# std_auc = np.std(aucs)
# ax.plot(
#     mean_fpr,
#     mean_tpr,
#     color="b",
#     label=r"Mean ROC (AUC = %0.2f $\pm$ %0.3f)" % (mean_auc, std_auc),
#     lw=2,
#     alpha=0.8,
# )
# std_tpr = np.std(tprs, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# ax.fill_between(
#     mean_fpr,
#     tprs_lower,
#     tprs_upper,
#     color="grey",
#     alpha=0.2,
#     label=r"$\pm$ 1 std. dev.",
# )
# ax.set(
#     xlim=[-0.05, 1.05],
#     ylim=[-0.05, 1.05],
#     title="AB Receiver operating characteristic ",
# )
# ax.legend(loc="lower right")
# plt.xlabel("1 - Specifity")
# plt.ylabel("Sensitivity")
# plt.show()
# # # # plt.savefig(
# # # #     'E:\\Spyder_2022.3.29\\output\\machinel\\lrq_output\\lrq_r\\AB Receiver operating characteristic.pdf')
# # # # # =============================================================================
# # # # # =============================================================================
# # # # # # rf 5折交叉ROC
# # # # # =============================================================================
# # # cv = StratifiedKFold(n_splits=5)
# # # tprs = []
# # # aucs = []
# # # mean_fpr = np.linspace(0, 1, 100)
# # # # fig = plt.figure(figsize=(2, 2), dpi=300)
# # # fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
# # # for i, (train, test) in enumerate(cv.split(X, y)):
# # #     Bag_model.fit(X[train], y[train])
# # #     viz = RocCurveDisplay.from_estimator(
# # #         Bag_model,
# # #         X[test],
# # #         y[test],
# # #         name="ROC fold {}".format(i),
# # #         alpha=0.3,
# # #         lw=1,
# # #         ax=ax,
# # #     )
# # #     interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
# # #     interp_tpr[0] = 0.0
# # #     tprs.append(interp_tpr)
# # #     aucs.append(viz.roc_auc)
# # # ax.plot([0, 1], [0, 1], linestyle="--", lw=2,
# # #         color="r", label="Reference", alpha=0.8)
# # # mean_tpr = np.mean(tprs, axis=0)
# # # mean_tpr[-1] = 1.0
# # # mean_auc = auc(mean_fpr, mean_tpr)
# # # std_auc = np.std(aucs)
# # # ax.plot(
# # #     mean_fpr,
# # #     mean_tpr,
# # #     color="b",
# # #     label=r"Mean ROC (AUC = %0.2f $\pm$ %0.3f)" % (mean_auc, std_auc),
# # #     lw=2,
# # #     alpha=0.8,
# # # )
# # # std_tpr = np.std(tprs, axis=0)
# # # tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# # # tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# # # ax.fill_between(
# # #     mean_fpr,
# # #     tprs_lower,
# # #     tprs_upper,
# # #     color="grey",
# # #     alpha=0.2,
# # #     label=r"$\pm$ 1 std. dev.",
# # # )
# # # ax.set(
# # #     xlim=[-0.05, 1.05],
# # #     ylim=[-0.05, 1.05],
# # #     title="BAG Receiver operating characteristic ",
# # # )
# # # ax.legend(loc="lower right")
# # # plt.xlabel("1 - Specifity")
# # # plt.ylabel("Sensitivity")
# # # plt.show()
# # # plt.savefig(
# # #     'E:\\Spyder_2022.3.29\\output\\machinel\\lrq_output\\lrq_r\\BAG Receiver operating characteristic.pdf')
# # # # # =============================================================================
# # # # # # gbm 5折交叉ROC
# # # # # =============================================================================
# cv = StratifiedKFold(n_splits=5)
# tprs = []
# aucs = []
# mean_fpr = np.linspace(0, 1, 100)
# # fig = plt.figure(figsize=(2, 2), dpi=300)
# fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
# for i, (train, test) in enumerate(cv.split(X, y)):
#     gbm.fit(X[train], y[train])
#     viz = RocCurveDisplay.from_estimator(
#         gbm,
#         X[test],
#         y[test],
#         name="ROC fold {}".format(i),
#         alpha=0.3,
#         lw=1,
#         ax=ax,
#     )
#     interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
#     interp_tpr[0] = 0.0
#     tprs.append(interp_tpr)
#     aucs.append(viz.roc_auc)
# ax.plot([0, 1], [0, 1], linestyle="--", lw=2,
#         color="r", label="Reference", alpha=0.8)
# mean_tpr = np.mean(tprs, axis=0)
# mean_tpr[-1] = 1.0
# mean_auc = auc(mean_fpr, mean_tpr)
# std_auc = np.std(aucs)
# ax.plot(
#     mean_fpr,
#     mean_tpr,
#     color="b",
#     label=r"Mean ROC (AUC = %0.2f $\pm$ %0.3f)" % (mean_auc, std_auc),
#     lw=2,
#     alpha=0.8,
# )
# std_tpr = np.std(tprs, axis=0)
# tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
# tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
# ax.fill_between(
#     mean_fpr,
#     tprs_lower,
#     tprs_upper,
#     color="grey",
#     alpha=0.2,
#     label=r"$\pm$ 1 std. dev.",
# )
# ax.set(
#     xlim=[-0.05, 1.05],
#     ylim=[-0.05, 1.05],
#     title="GBM Receiver operating characteristic ",
# )
# ax.legend(loc="lower right")
# plt.xlabel("1 - Specifity")
# plt.ylabel("Sensitivity")
# plt.show()
# plt.savefig(
# 'E:\\Spyder_2022.3.29\\output\\machinel\\lrq_output\\lrq_pre_r\\GBM Receiver operating characteristic.pdf')
# # # # # =============================================================================
# # # # # # mlp 5折交叉ROC
# # # # # =============================================================================
cv = StratifiedKFold(n_splits=5)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
# fig = plt.figure(figsize=(2, 2), dpi=300)
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
for i, (train, test) in enumerate(cv.split(X, y)):
    mlp.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        mlp,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
ax.plot([0, 1], [0, 1], linestyle="--", lw=2,
        color="r", label="Reference", alpha=0.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.3f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)
ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="MLP Receiver operating characteristic ",
)
ax.legend(loc="lower right")
plt.xlabel("1 - Specifity")
plt.ylabel("Sensitivity")
plt.show()
plt.savefig(
    'E:\\Spyder_2022.3.29\\output\\machinel\\lrq_output\\lrq_pre_r\\MLP Receiver operating characteristic.pdf')
# # # # # =============================================================================
# # # # # # xgb_model 5折交叉ROC
# # # # # =============================================================================
cv = StratifiedKFold(n_splits=5)
tprs = []
aucs = []
mean_fpr = np.linspace(0, 1, 100)
# fig = plt.figure(figsize=(2, 2), dpi=300)
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
for i, (train, test) in enumerate(cv.split(X, y)):
    xgb_model.fit(X[train], y[train])
    viz = RocCurveDisplay.from_estimator(
        xgb_model,
        X[test],
        y[test],
        name="ROC fold {}".format(i),
        alpha=0.3,
        lw=1,
        ax=ax,
    )
    interp_tpr = np.interp(mean_fpr, viz.fpr, viz.tpr)
    interp_tpr[0] = 0.0
    tprs.append(interp_tpr)
    aucs.append(viz.roc_auc)
ax.plot([0, 1], [0, 1], linestyle="--", lw=2,
        color="r", label="Reference", alpha=0.8)
mean_tpr = np.mean(tprs, axis=0)
mean_tpr[-1] = 1.0
mean_auc = auc(mean_fpr, mean_tpr)
std_auc = np.std(aucs)
ax.plot(
    mean_fpr,
    mean_tpr,
    color="b",
    label=r"Mean ROC (AUC = %0.2f $\pm$ %0.3f)" % (mean_auc, std_auc),
    lw=2,
    alpha=0.8,
)
std_tpr = np.std(tprs, axis=0)
tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
ax.fill_between(
    mean_fpr,
    tprs_lower,
    tprs_upper,
    color="grey",
    alpha=0.2,
    label=r"$\pm$ 1 std. dev.",
)
ax.set(
    xlim=[-0.05, 1.05],
    ylim=[-0.05, 1.05],
    title="XGB Receiver operating characteristic ",
)
ax.legend(loc="lower right")
plt.xlabel("1 - Specifity")
plt.ylabel("Sensitivity")
plt.show()
# # plt.savefig(
# #     'E:\\Spyder_2022.3.29\\output\\machinel\\lrq_output\\lrq_m\\XGBoost Receiver operating characteristic.pdf')
# # # # # =============================================================================
# # # # # 一张图多种机器学习方法的比较
# # # # # =============================================================================
# sns.set()
# random_state_new = 200
# X_train, X_test, y_train, y_test = train_test_split(
#     X, y, random_state=random_state_new)
AB_model.fit(X_train, y_train)
logis_model.fit(X_train, y_train)
mlp.fit(X_train, y_train)
Bag_model.fit(X_train, y_train)
gbm_model.fit(X_train, y_train)
xgb_model.fit(X_train, y_train)
fpr_ab, tpr_ab, thresholds_ab = roc_curve(
    y_test, AB_model.predict_proba(X_test)[:, 1])
fpr_lr, tpr_lr, thresholds_lr = roc_curve(
    y_test, logis_model.predict_proba(X_test)[:, 1])
fpr_bag, tpr_bag, thresholds_bag = roc_curve(
    y_test, Bag_model.predict_proba(X_test)[:, 1])
fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(
    y_test, mlp.predict_proba(X_test)[:, 1])
fpr_gbm, tpr_gbm, thresholds_gbm = roc_curve(
    y_test, gbm_model.predict_proba(X_test)[:, 1])
fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(
    y_test, xgb_model.predict_proba(X_test)[:, 1])
# fpr_ab, tpr_ab, thresholds_ab = roc_curve(
#     y, AB_model.predict_proba(X)[:, 1])
# fpr_lr, tpr_lr, thresholds_lr = roc_curve(
#     y, logis_model.predict_proba(X)[:, 1])
# fpr_bag, tpr_bag, thresholds_bag = roc_curve(
#     y, Bag_model.predict_proba(X)[:, 1])
# fpr_mlp, tpr_mlp, thresholds_mlp = roc_curve(
#     y, mlp.predict_proba(X)[:, 1])
# fpr_gbm, tpr_gbm, thresholds_gbm = roc_curve(
#     y, gbm_model.predict_proba(X)[:, 1])
# fpr_xgb, tpr_xgb, thresholds_xgb = roc_curve(
#     y, xgb_model.predict_proba(X)[:, 1])
# rf_prob = rf.predict_proba(X_test)
# lr_prob = rf.predict_proba(X_test)
roc_auc_ab = auc(fpr_ab, tpr_ab)
roc_auc_lr = auc(fpr_lr, tpr_lr)
roc_auc_bag = auc(fpr_bag, tpr_bag)
roc_auc_mlp = auc(fpr_mlp, tpr_mlp)
roc_auc_gbm = auc(fpr_gbm, tpr_gbm)
roc_auc_xgb = auc(fpr_xgb, tpr_xgb)
fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
plt.plot(fpr_lr, tpr_lr, label="ROC Curve LR; AUC = {:.3f}".format(roc_auc_lr))
plt.plot(fpr_ab, tpr_ab, label="ROC Curve AB; AUC = {:.3f}".format(roc_auc_ab))
plt.plot(fpr_bag, tpr_bag,
          label="ROC Curve BAG; AUC = {:.3f}".format(roc_auc_bag))
plt.plot(fpr_mlp, tpr_mlp,
          label="ROC Curve MLP; AUC = {:.3f}".format(roc_auc_mlp))
plt.plot(fpr_gbm, tpr_gbm,
          label="ROC Curve GBM; AUC = {:.3f}".format(roc_auc_gbm))
plt.plot(fpr_xgb, tpr_xgb,
          label="ROC Curve XGB; AUC = {:.3f}".format(roc_auc_xgb))
plt.plot([0, 1], [0, 1], color="navy", lw=2, linestyle="--", label='Reference')
plt.xlim([-0.05, 1.05])
plt.ylim([-0.05, 1.05])
plt.xlabel("1 - Specifity")
plt.ylabel("Sensitivity")
plt.title("ROC curve")
plt.legend(loc="lower right")
plt.show()
plt.savefig('E:\\Spyder_2022.3.29\\output\\machinel\\lrq_output\lrq_pre_r\\ROC curve.pdf')
# # # # # # =============================================================================
# # # # # # 计算AUC、精确率、灵敏度、特异度
# # # # # # =============================================================================
# # # #rf = ab
rf_score = AB_model.score(X_test, y_test)
lr_score = logis_model.score(X_test, y_test)
#bag = dt
dt_score = Bag_model.score(X_test, y_test)
mlp_score = mlp.score(X_test, y_test)
gbm_score = gbm_model.score(X_test, y_test)
xgb_score = xgb_model.score(X_test, y_test)
rf_prob = AB_model.predict(X_test)
lr_prob = logis_model.predict(X_test)
dt_prob = Bag_model.predict(X_test)
mlp_prob = mlp.predict(X_test)
gbm_prob = gbm_model.predict(X_test)
xgb_prob = xgb_model.predict(X_test)
# 混淆矩阵
rf_cf = confusion_matrix(y_test, rf_prob)
lr_cf = confusion_matrix(y_test, lr_prob)
dt_cf = confusion_matrix(y_test, dt_prob)
mlp_cf = confusion_matrix(y_test, mlp_prob)
gbm_cf = confusion_matrix(y_test, gbm_prob)
xgb_cf = confusion_matrix(y_test, xgb_prob)
rf_cf
lr_cf
dt_cf
mlp_cf
gbm_cf
xgb_cf
TN_rf, FP_rf, FN_rf, TP_rf = confusion_matrix(y_test, rf_prob).ravel()
TN_lr, FP_lr, FN_lr, TP_lr = confusion_matrix(y_test, lr_prob).ravel()
TN_dt, FP_dt, FN_dt, TP_dt = confusion_matrix(y_test, dt_prob).ravel()
TN_mlp, FP_mlp, FN_mlp, TP_mlp = confusion_matrix(y_test, mlp_prob).ravel()
TN_gbm, FP_gbm, FN_gbm, TP_gbm = confusion_matrix(y_test, gbm_prob).ravel()
TN_xgb, FP_xgb, FN_xgb, TP_xgb = confusion_matrix(y_test, xgb_prob).ravel()
sen_rf, spc_rf = round(TP_rf/(TP_rf+FN_rf), 3), round(TN_rf/(FP_rf+TN_rf), 3)
sen_lr, spc_lr = round(TP_lr/(TP_lr+FN_lr), 3), round(TN_lr/(FP_lr+TN_lr), 3)
sen_dt, spc_dt = round(TP_dt/(TP_dt+FN_dt), 3), round(TN_dt/(FP_dt+TN_dt), 3)
sen_mlp, spc_mlp = round(TP_mlp/(TP_mlp+FN_mlp),
                          3), round(TN_mlp/(FP_mlp+TN_mlp), 3)
sen_gbm, spc_gbm = round(TP_gbm/(TP_gbm+FN_gbm),
                          3), round(TN_gbm/(FP_gbm+TN_gbm), 3)
sen_xgb, spc_xgb = round(TP_xgb/(TP_xgb+FN_xgb),
                          3), round(TN_xgb/(FP_xgb+TN_xgb), 3)
AB_f1 = f1_score(y_test, rf_prob, average='macro')
LR_f1 = f1_score(y_test, lr_prob, average='macro')
DT_f1 = f1_score(y_test, dt_prob, average='macro')
MLP_f1 = f1_score(y_test, mlp_prob, average='macro')
GBM_f1 = f1_score(y_test, gbm_prob, average='macro')
XGB_f1 = f1_score(y_test, xgb_prob, average='macro')
print("AB的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(AB_f1, roc_auc_ab, rf_score, sen_rf, spc_rf))
print("LR的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(LR_f1, roc_auc_lr, lr_score, sen_lr, spc_lr))
print("BAG的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(DT_f1, roc_auc_bag, dt_score, sen_dt, spc_dt))
print("MLP的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(MLP_f1, roc_auc_mlp, mlp_score, sen_mlp, spc_mlp))
print("GBM的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(GBM_f1, roc_auc_gbm, gbm_score, sen_gbm, spc_gbm))
print("XGB的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
      .format(XGB_f1, roc_auc_xgb, xgb_score, sen_xgb, spc_xgb))
# # # roc_auc_rf, rf_score, sen_rf, spc_rf
# # # # # =============================================================================
# # # # # 绘制未平衡数据时的混淆矩阵
# # # # # =============================================================================
XGB_prob1 = xgb_model.predict(np.array(data))
cm = confusion_matrix(np.array(new_data[['M']]), XGB_prob1)
# fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NMO', 'MO'])
sns.set_style("white")
disp.plot(cmap='RdPu')
plt.title("Confusion Matrix of XGB")
plt.show()
plt.savefig('E:\\Spyder_2022.3.29\\output\\machinel\\lrq_output\\liver_cut_r\\Confusion Matrix of MLP.pdf')
# =============================================================================
# 绘制平衡数据后的混淆矩阵XGB
# =============================================================================
XGB_prob1 = xgb_model.predict(X)
cm = confusion_matrix(y, XGB_prob1)
# fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NMO', 'MO'])
sns.set_style("white")
disp.plot(cmap='RdPu')
plt.title("Confusion Matrix of XGB")
plt.show()
# =============================================================================
# 绘制平衡数据后的混淆矩阵AB
# =============================================================================
AB_prob1 = AB_model.predict(X)
cm = confusion_matrix(y, AB_prob1)
# fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NMO', 'MO'])
sns.set_style("white")
disp.plot(cmap='RdPu')
plt.title("Confusion Matrix of AB")
plt.show()
# =============================================================================
# 绘制平衡数据后的混淆矩阵LR
# =============================================================================
LR_prob1 = logis_model.predict(X)
cm = confusion_matrix(y, LR_prob1)
# fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NMO', 'MO'])
sns.set_style("white")
disp.plot(cmap='RdPu')
plt.title("Confusion Matrix of LR")
plt.show()
# =============================================================================
# 绘制平衡数据后的混淆矩阵GBM
# =============================================================================
GBM_prob1 = gbm_model.predict(X)
cm = confusion_matrix(y, GBM_prob1)
# fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NMO', 'MO'])
sns.set_style("white")
disp.plot(cmap='RdPu')
plt.title("Confusion Matrix of GBM")
plt.show()
# =============================================================================
# 绘制平衡数据后的混淆矩阵MLP
# =============================================================================
MLP_prob1 = mlp.predict(X)
cm = confusion_matrix(y, MLP_prob1)
# fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NMO', 'MO'])
sns.set_style("white")
disp.plot(cmap='RdPu')
plt.title("Confusion Matrix of MLP")
plt.show()
# =============================================================================
# 绘制平衡数据后的混淆矩阵MLP
# =============================================================================
BAG_prob1 = Bag_model.predict(X)
cm = confusion_matrix(y, BAG_prob1)
# fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['NMO', 'MO'])
sns.set_style("white")
disp.plot(cmap='RdPu')
plt.title("Confusion Matrix of BAG")
plt.show()
# # # # # =============================================================================
# # # # # # 变量重要性LR
# # # # # =============================================================================
# # # # lr.fit(X, y)
# # # # lr.predict(X)
# # # # # 输出模型系数
# # # # print('训练模型自变量参数为：', lr.coef_)
# # # # print('训练模型截距为：', lr.intercept_)
# # # # # 模型评价
# # # # print('模型的平均正确率为：', lr.score(X, y))
# # # # # 看下预测精度
# # # # y_predict = lr.predict(X)
# # # # accuracy_score(y, y_predict)
# # # # plt.rcParams['font.sans-serif'] = ['SimHei']  # 显示中文宋体
# # # # plt.rcParams['axes.unicode_minus'] = False  # 显示负号
# # # # coef_LR = pd.Series(lr.coef_.flatten()/max(lr.coef_.flatten())
# # # #                     * 100, index=X_data.columns, name='Var')
# # # # coef_lR_s = coef_LR.sort_values(ascending=False)[
# # # #     [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
# # # # coef_lR_s
# # # # plt.figure(figsize=(8, 8))
# # # # coef_lR_s.sort_values().plot(kind='barh', color='#30b2da')
# # # # plt.title("Feature Importances of LR")
# # # # plt.show()
# # # # plt.savefig(
# # # #     'E:\\Spyder_2022.3.29\\output\\machinel\\Feature Importances of LR.pdf')
# # # # # =============================================================================
# # # # # =============================================================================
# # # # # #RF
# # # # # =============================================================================
# # # # rf_importances = pd.Series(
# # # #     rf.feature_importances_, index=X_data.columns
# # # # ).sort_values(ascending=True).sort_values(ascending=False)[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
# # # # rf_importances
# # # # plt.figure(figsize=(8, 8))
# # # # rf_importances.sort_values().plot(kind='barh', color='#dad121')
# # # # plt.title("Feature Importances of RF")
# # # # plt.show()
# # # # # =============================================================================
# # # # # #DT
# # # # # =============================================================================
# # # # dt_importances = pd.Series(
# # # #     dt.feature_importances_, index=feature_names
# # # # ).sort_values(ascending=True).sort_values(ascending=False)[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
# # # # dt_importances
# # # # plt.figure(figsize=(8, 8))
# # # # dt_importances.sort_values().plot(kind='barh', color='#dad121')
# # # # plt.title("Feature Importances of DT")
# # # # plt.show()
# # # # # =============================================================================
# # # # # #MLP
# # # # # =============================================================================
# # # # mlp_importances = pd.Series(
# # # #     mlp.feature_importances_, index=X_data.columns
# # # # ).sort_values(ascending=True).sort_values(ascending=False)[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
# # # # mlp_importances
# # # # plt.figure(figsize=(8, 8))
# # # # mlp_importances.sort_values().plot(kind='barh', color='#dad121')
# # # # plt.title("Feature Importances of MLP")
# # # # plt.show()
# # # # # =============================================================================
# # # # # xgb_model 有用
# # # # # =============================================================================
# # # # gbm_model_importances = pd.Series(
# # # #     gbm_model.feature_importances_/max(gbm_model.feature_importances_)*100,
# # # #     index=X_data.columns).sort_values(ascending=False)[[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]]
# # # # gbm_model_importances
# # # # plt.figure(figsize=(8, 8))
# # # # gbm_model_importances.sort_values().plot(kind='barh', color='#fee494')
# # # # sns.set_style("darkgrid")
# # # # plt.title("Feature Importances of GBM ")
# # # # plt.xlim(0, 105)
# # # # plt.show()
# # # # plt.savefig(
# # # #     'E:\\Spyder_2022.3.29\\output\\machinel\\Feature Importances of XGB.pdf')
# # # # # =============================================================================
# # # # # 另外的shap模型
# # # # # =============================================================================
# # # # # 我们先训练好一个XGBoost model
# # # # model = xgboost.train({"learning_rate": 0.01},
# # # #                       xgboost.DMatrix(X, label=y), 100)
# # # # explainer = shap.TreeExplainer(model)
# # # # shap_values = explainer.shap_values(X)  # 传入特征矩阵X，计算SHAP值
# # # # # 可视化第一个prediction的解释   如果不想用JS,传入matplotlib=True
# # # # shap.force_plot(explainer.expected_value,
# # # #                 shap_values[0, :], X.iloc[0, :], matplotlib=True)
# # # # y_base = explainer.expected_value
# # # # print(y_base)
# # # # pred = model.predict(xgboost.DMatrix(X))
# # # # print(pred.mean())
# # # # shap.force_plot(explainer.expected_value, shap_values, X)
# # # # # summarize the effects of all the features
# # # # shap.summary_plot(shap_values, X, max_display=15)
# # # # shap.summary_plot(shap_values, X, plot_type="bar")
# # # # shap_interaction_values = explainer.shap_interaction_values(X)
# # # # shap.summary_plot(shap_interaction_values, X)
# # # # # =============================================================================
# # # # # 网页计算器绘制代码
# # # # # =============================================================================
# # # # # -*- coding: utf-8 -*-
# # # # """
# # # # Created on Mon Jun 27 09:00:07 2022
# # # # @author: Administrator
# # # # """
# # # # # 应用标题
# # # # st.set_page_config(page_title='Pred BM in PCa')
# # # # st.title('测试项目')
# # # # # conf
# # # # st.sidebar.markdown('## Variables')
# # # # BQL = st.sidebar.selectbox('BQL', ('No', 'Yes'), index=0)
# # # # SBSL = st.sidebar.selectbox('SBSL', ('No', 'Yes'), index=0)
# # # # HHD = st.sidebar.selectbox('HHD', ('No', 'Yes'), index=0)
# # # # FHH = st.sidebar.selectbox('FHH', ('No', 'Yes'), index=0)
# # # # diabetes = st.sidebar.selectbox('diabetes', ('No', 'Yes'), index=0)
# # # # memory = st.sidebar.selectbox('memory', ('No', 'Yes'), index=0)
# # # # HSD = st.sidebar.selectbox('HSD', ('No', 'Yes'), index=0)
# # # # PS = st.sidebar.selectbox('PS', ('No', 'Yes'), index=0)
# # # # EM = st.sidebar.selectbox('EM', ('No', 'Yes'), index=0)
# # # # stress = st.sidebar.selectbox('stress', ('No', 'Yes'), index=0)
# # # # gender = st.sidebar.selectbox('gender', ('male', 'female'), index=0)
# # # # age = st.sidebar.slider("age(years)", 15, 95, value=30, step=1)
# # # # BMI = st.sidebar.slider("BMI(单位)", 15.0, 40.0, value=20.0, step=0.1)
# # # # waistline = st.sidebar.slider("waistline(cm)", 50, 150, value=100, step=1)
# # # # NC = st.sidebar.slider("NC(单位)", 20.0, 60.0, value=30.0, step=0.1)
# # # # DrT = st.sidebar.slider("DrT(单位)", 0, 50, value=30, step=1)
# # # # SmT = st.sidebar.slider("SmT(单位)", 0, 50, value=30, step=1)
# # # # SmA = st.sidebar.slider("SmA(单位)", 0, 5, value=3, step=1)
# # # # SnT = st.sidebar.slider("SnT(单位)", 0, 50, value=30, step=1)
# # # # SuT = st.sidebar.slider("SuT(单位)", 0, 30, value=15, step=1)
# # # # ESS = st.sidebar.slider("ESS(单位)", 0, 25, value=10, step=1)
# # # # SBS = st.sidebar.slider("SBS(单位)", 0, 10, value=5, step=1)
# # # # AHI = st.sidebar.slider("AHI(单位)", 5.0, 150.0, value=50.0, step=0.1)
# # # # OAHI = st.sidebar.slider("OAHI(单位)", 0.0, 130.0, value=50.0, step=0.1)
# # # # minP = st.sidebar.slider("minP(单位)", 20, 100, value=50, step=1)
# # # # P90 = st.sidebar.slider("P90(单位)", 0.0, 100.0, value=70.0, step=0.1)
# # # # # 分割符号
# # # # st.sidebar.markdown('#  ')
# # # # st.sidebar.markdown('#  ')
# # # # st.sidebar.markdown('##### All rights reserved')
# # # # st.sidebar.markdown(
# # # #     '##### For communication and cooperation, please contact wshinana99@163.com, Wu Shi-Nan, Nanchang university')
# # # # # 传入数据
# # # # map = {'No': 0, 'Yes': 1, 'male': 0, 'female': 1}
# # # # # X_data = new_data[["BQL","SBSL","HHD","FHH","diabetes","memory","HSD","PS","EM","stress","gender"
# # # # #              ,"age","BMI","waistline","NC","DrT","SmT","SmA","SnT","SnT","ESS","SBS","AHI",
# # # # #              "OAHI","minP","P90"]]
# # # # BQL = map[BQL]
# # # # SBSL = map[SBSL]
# # # # HHD = map[HHD]
# # # # FHH = map[FHH]
# # # # diabetes = map[diabetes]
# # # # memory = map[memory]
# # # # HSD = map[HSD]
# # # # PS = map[PS]
# # # # EM = map[EM]
# # # # stress = map[stress]
# # # # gender = map[gender]
# # # # # age =map[age]
# # # # # BMI =map[BMI]
# # # # # waistline =map[waistline]
# # # # # NC =map[NC]
# # # # # DrT =map[DrT]
# # # # # SmT =map[SmT]
# # # # # SmA =map[SmA]
# # # # # SnT =map[SnT]
# # # # # SuT =map[SuT]
# # # # # ESS =map[ESS]
# # # # # SBS =map[SBS]
# # # # # AHI =map[AHI]
# # # # # OAHI =map[OAHI]
# # # # # minP =map[minP]
# # # # # P90 =map[P90]
# # # # # 数据读取，特征标注
# # # # hp_train = pd.read_csv('E:\\Spyder_2022.3.29\\data\\machinel\\data.csv')
# # # # hp_train['hypertension'] = hp_train['hypertension'].apply(
# # # #     lambda x: +1 if x == 1 else 0)
# # # # features = ["BQL", "SBSL", "HHD", "FHH", "diabetes", "memory", "HSD", "PS", "EM", "stress", "gender", "age", "BMI", "waistline", "NC", "DrT", "SmT", "SmA", "SnT", "SnT", "ESS", "SBS", "AHI",
# # # #             "OAHI", "minP", "P90"]
# # # # target = 'hypertension'
# # # # ros = RandomOverSampler(random_state=12, sampling_strategy='auto')
# # # # X_ros, y_ros = ros.fit_resample(hp_train[features], hp_train[target])
# # # # X_ros = np.array(X_ros)
# # # # XGB = XGBClassifier(n_estimators=360, max_depth=2,
# # # #                     learning_rate=0.1, random_state=0)
# # # # XGB.fit(X_ros, y_ros)
# # # # # 读存储的模型
# # # # # with open('XGB.pickle', 'rb') as f:
# # # # #    XGB = pickle.load(f)
# # # # sp = 0.5
# # # # # figure
# # # # is_t = (XGB.predict_proba(np.array([[BQL, SBSL, HHD, FHH, diabetes, memory, HSD, PS, EM, stress, gender, age, BMI, waistline, NC, DrT, SmT, SmA, SnT, SnT, ESS, SBS, AHI,
# # # #                                      OAHI, minP, P90]]))[0][1]) > sp
# # # # prob = (XGB.predict_proba(np.array([[BQL, SBSL, HHD, FHH, diabetes, memory, HSD, PS, EM, stress, gender, age, BMI, waistline, NC, DrT, SmT, SmA, SnT, SnT, ESS, SBS, AHI,
# # # #                                      OAHI, minP, P90]]))[0][1])*1000//1/10
# # # # if is_t:
# # # #     result = 'High Risk'
# # # # else:
# # # #     result = 'Low Risk'
# # # # if st.button('Predict'):
# # # #     st.markdown('## Risk grouping for hypertension:  '+str(result))
# # # #     if result == 'Low Risk':
# # # #         st.balloons()
# # # #     st.markdown('## Probability of High risk group:  '+str(prob)+'%')
# # # # # =============================================================================
# # # # # Feature importance的变量重要性
# # # # # =============================================================================
# sns.set()
# %matplotlib
# explainer = shap.LinearExplainer(logis_model, X_data, feature_dependence="independent")
# shap_values = explainer.shap_values(X_data)  # 传入特征矩阵X，计算SHAP值
# #Feature importances1 and 2
# a =76
# shap.initjs()
# plot1 = shap.force_plot(explainer.expected_value,
#                 shap_values[a, :], 
#                 X_data.iloc[a, :], 
#                 figsize=(15, 5),
#                 link = "logit",
#                 matplotlib=True,
#                 out_names = "Output value")
# # #Feature importances
# sns.set()
# shap.summary_plot(shap_values, 
#                   X_data,
#                   plot_type="violin", 
#                   max_display=10,
#                   color='#fee494')
# #柱状图
# shap.summary_plot(shap_values, X_data, plot_type="bar")
# # =============================================================================
# # Table 3表格的制作
# # =============================================================================
# print("AB的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
#       .format(AB_f1, roc_auc_ab, rf_score, sen_rf, spc_rf))
# print("LR的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
#       .format(LR_f1, roc_auc_lr, lr_score, sen_lr, spc_lr))
# print("BAG的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
#       .format(DT_f1, roc_auc_bag, dt_score, sen_dt, spc_dt))
# print("MLP的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
#       .format(MLP_f1, roc_auc_mlp, mlp_score, sen_mlp, spc_mlp))
# print("GBM的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
#       .format(GBM_f1, roc_auc_gbm, gbm_score, sen_gbm, spc_gbm))
# print("XGB的F1值为：{:.3f}；AUC值为：{:.3f}；精确率为：{:.3f}；灵敏度为：{:.3f}；特异度为：{:.3f}"
#       .format(XGB_f1, roc_auc_xgb, xgb_score, sen_xgb, spc_xgb))
# =============================================================================
# 绘制条形图的重要性排序
# =============================================================================
explainer = shap.Explainer(xgb_model, data)
shap_values = explainer.shap_values(data, check_additivity = False)  # 传入特征矩阵X，计算SHAP值
# #Feature importances
sns.set()
shap.summary_plot(shap_values, 
                  data,
                  plot_type="violin", 
                  max_display=6,
                  color='#fee494',
                  title='Feature importance')
# =============================================================================
# #Feature importances1 and 2
# =============================================================================
explainer = shap.Explainer(xgb_model, data)
shap_values = explainer.shap_values(data, check_additivity = False)  # 传入特征矩阵X，计算SHAP值
a = 1485
shap.initjs()
plot1 = shap.force_plot(explainer.expected_value,
                shap_values[a, :], 
                X_data.iloc[a, :], 
                figsize=(15, 5),
                # link = "logit",
                matplotlib=True,
                out_names = "Output value")
# =============================================================================
# #柱状图
# =============================================================================
shap.summary_plot(shap_values, data, plot_type="bar")
