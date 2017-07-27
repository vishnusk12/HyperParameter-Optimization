# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 11:09:23 2017

@author: Impact
"""

import numpy as np
from scipy.stats import uniform as sp_rand
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, BaggingClassifier, ExtraTreesClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import SVC
from xgboost.sklearn import XGBClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint as sp_randint


criterion_list_TREE = ['gini', 'entropy']
splitter_TREE = ['best', 'random']
loss_list_SGD = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
penalty_list_SGD = ['none', 'l2', 'l1', 'elasticnet']
penalty_LOG = ['l1', 'l2']
solver_LOG = ['newton-cg', 'lbfgs', 'liblinear', 'sag']
multi_class_LOG = ['ovr', 'multinomial']
criterion_RF = ['gini', 'entropy']
min_samples_split_TRF = range(2,11)
max_samples_BAG = range(1,11)
max_features_TRF = range(1,11)
activation_MLP = ['identity', 'logistic', 'tanh', 'relu']
solver_MLP = ['lbfgs', 'sgd', 'adam']
learning_rate_MLP = ['constant', 'invscaling', 'adaptive']
kernel_list_SVM = ['rbf', 'poly', 'sigmoid', 'linear']
loss_GB = ['deviance', 'exponential']
criterion_GB = ['friedman_mse', 'mse']
algorithm_ADA = ['SAMME', 'SAMME.R']
C_XG = range(1,11)
solver_LDA = ['svd', 'lsqr', 'eigen']


def Grid_classifier(a, b):
    N_range_KNN = range(1,31)
    C_LOGSVMXG = 10. ** np.arange(-3, 8)
    n_estimators_RF = [10, 100, 1000, 10000]
    hidden_layer_sizes_MLP = [100, 1000, 10000]
    gamma_range_Grid_SVM = 10. ** np.arange(-5, 4)
    alpha_NB = range(1,11)
    
    params_Grid_KNN = dict(n_neighbors=N_range_KNN)
    params_Grid_tree = dict(criterion=criterion_list_TREE, splitter=splitter_TREE, min_samples_split=min_samples_split_TRF, max_features=max_features_TRF)
    params_Grid_SGD = dict(loss=loss_list_SGD, penalty=penalty_list_SGD)
    params_Grid_LOG = dict(penalty=penalty_LOG, C=C_LOGSVMXG, solver=solver_LOG, multi_class=multi_class_LOG)
    params_Grid_RF = dict(n_estimators=n_estimators_RF, criterion=criterion_RF, min_samples_split=min_samples_split_TRF, max_features=max_features_TRF)
    params_Grid_MLP = dict(hidden_layer_sizes=hidden_layer_sizes_MLP, activation=activation_MLP, solver=solver_MLP, learning_rate=learning_rate_MLP)
    params_Grid_SVM = dict(gamma=gamma_range_Grid_SVM, C=C_LOGSVMXG, kernel=kernel_list_SVM)
    params_Grid_NB = dict(alpha=alpha_NB)
    params_Grid_GB = dict(loss=loss_GB, criterion=criterion_GB, min_samples_split=min_samples_split_TRF)
    params_Grid_ADA = dict(algorithm=algorithm_ADA)
    params_Grid_XG = dict(C=C_LOGSVMXG)
    params_Grid_BAG = dict(max_samples=max_samples_BAG, max_features=max_features_TRF)
    params_Grid_LDA = dict(solver=solver_LDA)
    params_Grid_ET = dict(criterion=criterion_list_TREE, max_features=max_features_TRF, min_samples_split=min_samples_split_TRF)
    
    Grid_KNN = GridSearchCV(KNeighborsClassifier(), params_Grid_KNN)
    Grid_tree = GridSearchCV(DecisionTreeClassifier(), params_Grid_tree)
    Grid_SGD = GridSearchCV(SGDClassifier(), params_Grid_SGD)
    Grid_LOG = GridSearchCV(LogisticRegression(), params_Grid_LOG)
    Grid_RF = GridSearchCV(RandomForestClassifier(), params_Grid_RF)
    Grid_MLP = GridSearchCV(MLPClassifier(), params_Grid_MLP)
    Grid_SVM = GridSearchCV(SVC(), params_Grid_SVM)
    Grid_NB = GridSearchCV(BernoulliNB(), params_Grid_NB)
    Grid_GB = GridSearchCV(GradientBoostingClassifier(), params_Grid_GB)
    Grid_ADA = GridSearchCV(AdaBoostClassifier(), params_Grid_ADA)
    Grid_XG = GridSearchCV(XGBClassifier(), params_Grid_XG)
    Grid_BAG = GridSearchCV(BaggingClassifier(), params_Grid_BAG)
    Grid_LDA = GridSearchCV(LinearDiscriminantAnalysis(), params_Grid_LDA)
    Grid_ET = GridSearchCV(ExtraTreesClassifier(), params_Grid_ET)
    
    Grids = [Grid_KNN, Grid_tree, Grid_SGD, Grid_LOG, Grid_RF, Grid_MLP, Grid_SVM, Grid_NB, Grid_GB, Grid_ADA, Grid_XG, Grid_BAG, Grid_LDA, Grid_ET]
    
    list_Grid = []
    for grid in Grids:
        try:
            dict_Grid = {}
            grid.fit(a, b)
            dict_Grid['Best_Estimator'] = grid.best_estimator_
            dict_Grid['Accuracy'] = grid.best_score_
            list_Grid.append(dict_Grid)
        except:
            pass
    
    Best_Classifier_Grid = max(list_Grid, key=lambda x:x['Accuracy'])
    return Best_Classifier_Grid

def Random_classifier(a, b):
    N_range_KNN = sp_randint(1, 31)
    C_LOGSVM = sp_rand()
    n_estimators_RF = sp_randint(10, 10000)
    hidden_layer_sizes_MLP = sp_randint(100, 1000)
    gamma_range_Random_SVM = sp_rand()
    alpha_NB = sp_rand()
    
    params_Random_KNN = dict(n_neighbors=N_range_KNN)
    params_Random_tree = dict(criterion=criterion_list_TREE, splitter=splitter_TREE, min_samples_split=min_samples_split_TRF, max_features=max_features_TRF)
    params_Random_SGD = dict(loss=loss_list_SGD, penalty=penalty_list_SGD)
    params_Random_LOG = dict(penalty=penalty_LOG, C=C_LOGSVM, solver=solver_LOG, multi_class=multi_class_LOG)
    params_Random_RF = dict(n_estimators=n_estimators_RF, criterion=criterion_RF, min_samples_split=min_samples_split_TRF, max_features=max_features_TRF)
    params_Random_MLP = dict(hidden_layer_sizes=hidden_layer_sizes_MLP, activation=activation_MLP, solver=solver_MLP, learning_rate=learning_rate_MLP)
    params_Random_SVM = dict(gamma=gamma_range_Random_SVM, C=C_LOGSVM, kernel=kernel_list_SVM)
    params_Random_NB = dict(alpha=alpha_NB)
    params_Random_GB = dict(loss=loss_GB, criterion=criterion_GB, min_samples_split=min_samples_split_TRF)
    params_Random_ADA = dict(algorithm=algorithm_ADA)
    params_Random_XG = dict(C=C_XG)
    params_Random_BAG = dict(max_samples=max_samples_BAG, max_features=max_features_TRF)
    params_Random_LDA = dict(solver=solver_LDA)
    params_Random_ET = dict(criterion=criterion_list_TREE, max_features=max_features_TRF, min_samples_split=min_samples_split_TRF)

    Random_KNN = RandomizedSearchCV(KNeighborsClassifier(), params_Random_KNN)
    Random_tree = RandomizedSearchCV(DecisionTreeClassifier(), params_Random_tree)
    Random_SGD = RandomizedSearchCV(SGDClassifier(), params_Random_SGD)
    Random_LOG = RandomizedSearchCV(LogisticRegression(), params_Random_LOG)
    Random_RF = RandomizedSearchCV(RandomForestClassifier(), params_Random_RF)
    Random_MLP = RandomizedSearchCV(MLPClassifier(), params_Random_MLP)
    Random_SVM = RandomizedSearchCV(SVC(), params_Random_SVM)
    Random_NB = RandomizedSearchCV(BernoulliNB(), params_Random_NB)
    Random_GB = GridSearchCV(GradientBoostingClassifier(), params_Random_GB)
    Random_ADA = GridSearchCV(AdaBoostClassifier(), params_Random_ADA)
    Random_XG = GridSearchCV(XGBClassifier(), params_Random_XG)
    Random_BAG = GridSearchCV(BaggingClassifier(), params_Random_BAG)
    Random_LDA = GridSearchCV(LinearDiscriminantAnalysis(), params_Random_LDA)
    Random_ET = GridSearchCV(ExtraTreesClassifier(), params_Random_ET)

    Randoms = [Random_KNN, Random_tree, Random_SGD, Random_LOG, Random_RF, Random_MLP, Random_SVM, Random_NB, Random_GB, Random_ADA, Random_XG, Random_BAG, Random_LDA, Random_ET]
    
    list_Random = []
    for ran in Randoms:
        try:
            dict_Random = {}
            ran.fit(a, b)
            dict_Random['Best_Estimator'] = ran.best_estimator_
            dict_Random['Accuracy'] = ran.best_score_
            list_Random.append(dict_Random)
        except:
            pass
    
    Best_Classifier_Random = max(list_Random, key=lambda x:x['Accuracy'])
    return Best_Classifier_Random
