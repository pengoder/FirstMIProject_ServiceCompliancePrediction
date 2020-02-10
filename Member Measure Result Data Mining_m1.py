# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 13:19:54 2017

@author: Peng
"""
#%load_ext autoreload # Reload all modules (except those excluded by %aimport) automatically now.
#%autoreload 2 # Reload all modules (except those excluded by %aimport) every time before executing the Python code typed.

import pandas as pd
import numpy as np
from sklearn import preprocessing, metrics
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import RFE, RFECV
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
import numpy.core.multiarray
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib.style as style
from yellowbrick.classifier import ClassificationReport, ROCAUC
import seaborn as sns
import os
from numpy import nanmean
from collections import Counter
import pickle
import sys

style.use('ggplot')

def read_data(load_path):
    df = pd.read_csv(load_path, ',')
    return df

def descriptive_analysis(df):
#    sns.distplot(df.PCP_MEM_CNT, kde=True)
#    df.PCP_MEM_CNT.value_counts()
#    plt.hist(df.TTL_COST, bins=100)
#    plt.show()
#    df.boxplot(column='TTL_LIABILITY', by='HEDIS_POPULATION')
    pd.tools.plotting.scatter_matrix(df, alpha=0.2, diagonal='kde')
    
def fillna_df(df):
    """ Transform dataset
    ->scale some fields
    """
    df.fillna(0, inplace=True)
#    df['MEM_GNDR_CD'] = np.where(df['MEM_GNDR_CD'] == 'M', 1, 0)
#    to_scale = conf.to_scale
#    scaler = preprocessing.StandardScaler()
#    df[to_scale] = scaler.fit_transform(df[to_scale])
#    df.drop(df(['PCP_MEM_CNT'] < 50), inplace=True)
    
def transformer():
    """This is based on correlation heatmap
    """
    preprocessing.StandardScaler()

def plot_difference_before_after_sacling(x_var):
    fig, ax1 = plt.subplots(ncols=1, figsize=(6, 5))
    ax1.set_title('Before Scaling')
    sns.kdeplot(x_var['TTL_COST'], ax=ax1)
    sns.distplot(x_var['TTL_COST'], kde=True/False)
    plt.hist(x_var['TTL_COST'], bins=200, normed=True, cumulative=True)
    sns.kdeplot(x_var['TTL_LIABILITY'], ax=ax1)
    sns.kdeplot(x_var['HH_INC_AVG'], ax=ax1)
#    ax2.set_title('After Scaling')
#    sns.kdeplot(some_change[0], ax=ax2)
#    sns.kdeplot(some_change[1], ax=ax2)
#    sns.kdeplot(some_change[2], ax=ax2)
    plt.show()
    
def chk_del_field_same_val(x_var):
    """ to check if any field only contains same single value; then delete this field from data set
    """
    to_delete = []
    for col in x_var.columns:
        if x_var[col].nunique() == 1:
            to_delete.append(col)
    print (to_delete)
    x_var.drop(to_delete, axis=1, inplace=True)
    return x_var
    
def slice_data(df, hedis_population, msr_desc):
    """ Slice data by
        ->hedis population
        ->measure
        ->pcp size
        Select influential feasures for the model
    """

    # for different population
    if hedis_population == '':
        df_slice = df[df.MSR_DESC==msr_desc]
    else:
        df_slice = df[(df.HEDIS_POPULATION==hedis_population) & (df.MSR_DESC==msr_desc)]
#    df_slice = df.query("HEDIS_POPULATION = @hedis_population and MSR_DESC = @msr_desc")
    feature_list = conf.feature_list
    x_var = df_slice.loc[:, feature_list]
    y_var = df_slice['ADMIN_NUMERATOR']
    fillna_df(x_var)
    x_var = chk_del_field_same_val(x_var)
    return x_var, y_var

def resample_imbalanced_data(X, y, resample_method):
    """using sampling techniques to resize imbalance data set
    """
    from imblearn.over_sampling import SMOTE, RandomOverSampler
    from imblearn.under_sampling import RandomUnderSampler
    from sklearn.utils import resample
    
    print ('Original dataset shape {}'.format(Counter(y)))
    ######
    ### oversample minority class
    if resample_method == 'Oversampling':
        df_slice = X.join(y.to_frame())
        if (y==1).sum() > (y==0).sum():
            cls_majority = 1
            cls_minority = 0
        else:
            cls_majority = 0
            cls_minority = 1
        n_samples = (y==cls_majority).sum()
        df_majority = df_slice[df_slice['ADMIN_NUMERATOR'] == cls_majority]
        df_minority = df_slice[df_slice['ADMIN_NUMERATOR'] == cls_minority]
        df_minority_upsampled = resample(df_minority,
                                        replace=True
                                        , # sample with replacement
                                        n_samples=n_samples,# to match majority class
                                        random_state=123 # reproducible results
                                        ) 
        # combine majority class with upsampled minority class
        df_upsampled = pd.concat([df_majority, df_minority_upsampled])
        feature_list = conf.feature_list
        x_res, y_res = df_upsampled[feature_list], df_upsampled['ADMIN_NUMERATOR']
    #####
    elif resample_method == 'RandomOverSampler':
        sm = RandomOverSampler()
        x_res, y_res = sm.fit_sample(X, y)
    ### SMOTE to do data augmentation
    elif resample_method == 'SMOTE':
        sm = SMOTE()
        x_res, y_res = sm.fit_sample(X, y)
    #####
    elif resample_method == 'RandomUnderSampler':
        sm = RandomUnderSampler()
        x_res, y_res = sm.fit_sample(X, y)
    print ('Resampled dataset shape {}'.format(Counter(y_res)))
    return x_res, y_res
    
#feature selection
def feature_selection(X, y, model):
    feature_list = X.columns
#    rfe = RFE(model, 9)
    rfe = RFECV(estimator=model, cv=10, scoring='accuracy')
    rfecv = rfe.fit(X, y)
    print ('Optimal number of features:  {}'.format(rfecv.n_features_))
    af = rfecv.ranking_
    ftr_rnk = pd.DataFrame(np.column_stack((feature_list ,list(af))), columns=['COL_NM', 'RNK'])
    # Plot number of features VS. cross-validation scores
    plt.figure()
    plt.xlabel('Number of feature selected')
    plt.ylabel('Cross validation score (no. of positive prediction in all positive records)')
    plt.plot(range(1, len(rfecv.grid_scores_) + 1), rfecv.grid_scores_)
    plt.show()
    return ftr_rnk, rfecv.n_features_

def feature_join(X, y, lst_model):
    """this function is specifically used to get features shared by different models
    """
    def feature_select(X, y, model):
        feature_list = X.columns
        rfe = RFE(estimator=model, n_features_to_select=10)
        rfe.fit(X, y)
        af = rfe.ranking_
        ftr_rnk = ftr_rnk = pd.DataFrame(np.column_stack((feature_list ,list(af))), columns=['COL_NM', 'RNK'])
        return ftr_rnk
    feature_decided = pd.DataFrame()
    for mdl in lst_model:
        feature = feature_select(X, y, mdl)
        feature_selected = feature[feature['RNK'] == 1]
        print (feature_selected)
        if feature_decided.empty:
            feature_decided = feature_selected
        else:
            feature_decided = feature_decided.merge(feature_selected, left_on='COL_NM', right_on='COL_NM')
    return feature_decided
    
#Detect multicollinearity in independent variables
def detect_multicollinearity(X):
    corr = np.corrcoef(X, rowvar=0)
    where_are_NaNs = np.isnan(corr)
    corr[where_are_NaNs] = 0
    f, ax = plt.subplots(figsize=(18, 18))
    sns.heatmap(X.corr(), annot=True, linewidths=.5, fmt='.1f', ax=ax)
    w, v = np.linalg.eig(corr)
    return pd.DataFrame(np.column_stack((X.columns, list(w)))), v

def remove_field_correlation(X_in, f_to_rmv):
    X_in.drop(f_to_rmv, axis=1, inplace=True)

#decomposition
def decompositionPCA(X):
    def doPCA(data):
        from sklearn.decomposition import PCA
        pca = PCA(n_components = 15)
        pca.fit(data)
        return pca
    pca = doPCA(X)
    print (pca.explained_variance_ratio_)
    print (pca.explained_variance_ratio_.cumsum())
    # plot explained variance to see the elbow
    plt.plot(pca.explained_variance_ratio_)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()
#    first_pc = pca.components_[0]
#    second_pc = pca.components_[1]
    
def plot_pca_deminstions(X):
    """ plot projections on Principal Components plane
        this is good to visualize a 2D or 3D PC
    """
    pca = PCA(n_components=3).fit(X)
    reduced_data = pca.transform(X)
    reduced_data = pd.DataFrame(reduced_data, columns = ['Dimension 1', 'Dimension 2', 'Dimension 3'])

    def biplot(data, reduced_data, pca):
        from mpl_toolkits.mplot3d import Axes3D
    #                fig, ax = plt.subplots(figsize = (14,8))
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        #scatterplot of the reduced data
        ax.scatter(xs=reduced_data.loc[:, 'Dimension 1'], ys=reduced_data.loc[:, 'Dimension 2'], zs = reduced_data.loc[:, 'Dimension 3'], facecolors='b', edgecolors='b', s=70, alpha=0.5)
        feature_vectors = pca.components_.T
        # using scaling factors to make the arrows
        arrow_size, text_pos = 1.0, 2.0
        
        # projections of the original features
        for i, v in enumerate(feature_vectors):
            ax.arrow(0, 0, arrow_size*v[0], arrow_size*v[1], arrow_size*v[2], head_width=0.2, head_length=0.2, linewidth=2, color='red')
            ax.text(v[0]*text_pos, v[1]*text_pos, v[2]*text_pos, data.columns[i], color='black', ha='center', va='center', fontsize=18)
            print (i)
        ax.set_xlabel("Dimension 1", fontsize=14)
        ax.set_ylabel("Dimension 2", fontsize=14)
        ax.set_zlabel("Dimension 3", fontsize=14)
        ax.set_title("PC plane with original feature projections.", fontsize=16)
        return ax
    
    biplot(X, reduced_data, pca)


def statistical_regression(X, y):
    import statsmodels.api as sm
#    from statsmodels.sandbox.stats.multicomp import fdrcorrection0
    logit_feature = conf.in_logit_model
    X_logit = X[logit_feature]
    logit_model = sm.Logit(y, X_logit)
    result = logit_model.fit()
    print (result.summary())
    print (np.exp(result.params))
        
def plot_roc_curve(expected, predicted, cclf):
    false_positive_rate, true_positive_rate, thresholds = metrics.roc_curve(expected, predicted)
#    roc_auc = metrics.auc(false_positive_rate, true_positive_rate)
    roc_auc = metrics.roc_auc_score(expected, predicted)
    plt.title('Receiver Operating Characteristic')
    plt.plot(false_positive_rate, true_positive_rate, 'b', label='AUC=%0.2f' %roc_auc)
    plt.legend(loc='lower right')
    plt.plot([0,1], [0,1], 'r--')
    plt.xlim([-0.1, 1.2])
    plt.ylim([-0.1, 1.2])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.show()
        
def plot_precision_recall_curve(expected, predicted):
    from sklearn.metrics import precision_recall_curve
    from sklearn.metrics import average_precision_score
    
    average_precision = average_precision_score(expected, predicted)
    precision, recall, _ = precision_recall_curve(expected, predicted)
    plt.step(recall, precision, color='b', alpha=0.2,
             where='post')
    plt.fill_between(recall, precision, step='post', alpha=0.2,
                     color='b')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
    plt.show()
    
def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Greys):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    import itertools
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)
    
    plt.grid(False)
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

#    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    style.use('ggplot')
    
def cross_validation_ML(X, y, ML_name, MLAlgrm, threshold, n_feature=10, resample_method=None, re_sample=None):
    """Use ML Logistic Regression in Sci-kit Learn to fit the model
       ->split to training set and testing set
    """
    def make_pipeline(model):
        from sklearn.pipeline import FeatureUnion
        from sklearn.pipeline import Pipeline
        # create feature union
        features = []
#        features.append(('pca', PCA(n_components=3)))
        features.append(('transformer', preprocessing.StandardScaler()))
        features.append(('rfe', RFE(MLAlgrm, n_feature)))
        feature_union = FeatureUnion(features)        
        # create pipeline
        estimators = []
        estimators.append(('feature_union', feature_union))
        estimators.append(('model', model))
        print (estimators)
        clf = Pipeline(estimators)
        return clf
        
    def calculate_probs(X_train, y_train, X_test, base_estimator, threshold):
        """this is using calibration to get predicted probability
        """
        from sklearn.calibration import CalibratedClassifierCV
        # with pipeline
        estimators = make_pipeline(base_estimator)
        # without pipeline
#        estimators = base_estimator
        if ML_name in ('MLP', 'KNN', 'Voting'): # no pipeline
            cclf = CalibratedClassifierCV(base_estimator=base_estimator, cv=5)
        else:
            cclf = CalibratedClassifierCV(base_estimator=estimators, cv=5)
        cclf.fit(X_train, y_train)
        res = cclf.predict_proba(X_test)[:, 1]
        predicted = np.array([1 if x>=threshold else 0 for x in res])
#        pred_prob = cclf.predict(X_test)
        return res, predicted, cclf
        
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)
    if re_sample == 1:
        X_train, y_train = resample_imbalanced_data(X_train, y_train, resample_method)
    preds, predicted, cclf = calculate_probs(X_train, y_train, X_test, MLAlgrm, threshold)
    print (cclf)
    expected = y_test
    print('The model''s acuracy is {}'.format(metrics.accuracy_score(expected, predicted)))
#    print(metrics.roc_curve(expected, predicted))
    print (metrics.classification_report(expected, predicted))
    cm = metrics.confusion_matrix(expected, predicted)
    class_names = ['Not Compliant', 'Compliant']
    plt.figure()
    plot_confusion_matrix(cm, class_names)
    
#    f, ax1 = plt.subplots(figsize=(3, 3))
#    sns.heatmap(cm, annot=True, fmt='d', ax=ax1)
    # plot ROC curve
    plt.figure()
    plot_roc_curve(expected, preds, MLAlgrm)
    # plot precision/recall curve
    plot_precision_recall_curve(expected, preds)
    # plot distribution of probability of prediction
    sns.distplot(preds, kde=True)
    ### save this algorithm
    return cclf
    
def predict_new_dataset(X, y, cclf, threshold):
#    predicted = cclf.predict(X)
    preds = cclf.predict_proba(X)[:, 1]
    predicted = np.array([1 if x>=threshold else 0 for x in preds])
    expected = y
    print (metrics.accuracy_score(expected, predicted))
    print (metrics.classification_report(expected, predicted))
    print (metrics.confusion_matrix(expected, predicted))
    rate_expected = (y == 1).sum()/len(y)
    rate_predicted = (predicted == 1).sum()/len(predicted)
    print ('The expected(true) rate is {}, and the predicted rate is {}'.format(rate_expected, rate_predicted))
    plot_roc_curve(expected, preds, cclf)
    plot_precision_recall_curve(expected, preds)
    
    
###########
###########
### MODEL FITTING
###########
###########
hedis_population = 'Medicare Adv HMO POS'
msr_desc = 'ABA'


rootPath = r'H:\BU2016\Folder\Project\015_Model Building\Gap Close Prediction - based on Member Profile'
os.chdir(rootPath)
import conf

load_path = 'Member Profile_Selected.txt'
df = read_data(load_path)
x_var, y_var = slice_data(df, hedis_population, msr_desc)
print ('Denominator is {}, numerator is {}'.format(len(y_var), (y_var==1).sum()))
w, v = detect_multicollinearity(x_var)
### 
#l_to_rmv = ['TTL_COST']
#remove_field_correlation(x_var, l_to_rmv)
#descriptive_analysis(x_var)

#decompositionPCA(x_var)
#plot_pca_deminstions(x_var)

#### Standard Statistical Analysis, Logistic Regression
#************
statistical_regression(x_var, y_var)
#************
####

#### Machine Learning
# Various algorithms selected
#*********
cclf1 = LinearSVC(C=2,
                  dual=False,
                  class_weight='balanced' # penalize
                  )
cclf2 = KNeighborsClassifier(n_jobs=4)
cclf3 = RandomForestClassifier()
cclf4 = AdaBoostClassifier()
cclf5 = LogisticRegression()
cclf6 = MLPClassifier()
eclf = VotingClassifier(estimators=[('Logit', cclf5), ('rf', cclf3), ('Adab', cclf4)], # cannot use RFECV to select features 
                                    voting='soft',                                                  # because it doesn't have "coef_" or "feature_importances_"
                                    weights=[4, 4, 2])
dict_cclf = {'LSVC': cclf1,
             'KNN': cclf2,
             'RF': cclf3,
             'AdaB': cclf4,
             'Logit': cclf5,
             'MLP': cclf6,
             'Voting': eclf
             }
feature_shared = feature_join(x_var, y_var, [dict_cclf['LSVC'], dict_cclf['RF']])
af, n_feature = feature_selection(x_var, y_var, dict_cclf['LSVC'])
threshold = 0.5
classifier = cross_validation_ML(x_var, y_var, 'LSVC', dict_cclf['LSVC'], threshold, n_feature, 'SMOTE', 0)

#file_name = 'Trained Classifier.sav'
#pickle.dump(classifier, open(file_name, 'wb'))
x_var.drop('M_85', axis=1, inplace=True)
###########
###########
### PREDICTION WITH NEW DATA
###########
###########
new_dataset_path = 'New Member Profile.txt'
df_new = read_data(new_dataset_path)
x_new, y_new = slice_data(df_new, hedis_population, msr_desc)
print ('Denominator is {}, numerator is {}'.format(len(y_new), (y_new==1).sum()))
#pickle_in = open (file_name, 'rb')
#trained_model = pickle.load(pickle_in)
select_feature = x_var.columns.tolist()
predict_new_dataset(x_new[select_feature], y_new, classifier, threshold)

###########
###########
### RESULTS SECTION
###########
###########

rl = [
'PCP_CARE_ABA',
'PCP_CARE_ART',
'PCP_CARE_BCS',
'PCP_CARE_COL',
'PCP_CARE_CDC_A1C',
'PCP_CARE_CDC_EYE',
'PCP_CARE_CDC_NEPH',
'PCP_CARE_OMW',
'IPA_CARE_ABA',
'IPA_CARE_ART',
'IPA_CARE_BCS',
'IPA_CARE_COL',
'IPA_CARE_CDC_A1C',
'IPA_CARE_CDC_EYE',
'IPA_CARE_CDC_NEPH',
'IPA_CARE_OMW'
]
abb = df.loc[:, rl]
sns.heatmap(abb.corr(), annot=True, linewidths=.5, fmt='.1f')
del abb, rl
