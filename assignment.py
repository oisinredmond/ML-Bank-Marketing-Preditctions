# Oisin Redmond - C15492202 - DT228/4
# Machine Learning Assignment 2

import numpy as np
import pandas as pd
import csv
import matplotlib.pyplot as plt
from pandas.plotting import scatter_matrix
from sklearn import model_selection
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC


trainingset = pd.read_csv('data/trainingset.txt', index_col=0)
queryset = pd.read_csv('data/queries.txt', index_col=0)


trainingset.columns = ['age', 'job', 'marital', 'education', 'default', 'balance', 'housing', 'loan', 'contact', 'day',
                       'month', 'duration', 'campaign', 'pdays', 'previous', 'poutcome', 'y']


print(trainingset.apply(lambda x: sum(x.isnull()),axis=0))

count = trainingset.groupby('y').size()
percent = count/len(trainingset)*100
print(percent)

queryset.columns = trainingset.columns

queries_index = queryset.index

numeric_features = ['age', 'balance', 'day', 'duration',  'campaign', 'pdays', 'previous']

categorical_features = ['job', 'marital', 'education', 'default', 'housing', 'loan','month', 'contact', 'poutcome']

numeric_data = pd.DataFrame(trainingset, columns=numeric_features)
categorical_data = pd.DataFrame(trainingset, columns=categorical_features)

temp = numeric_data.describe()
#temp.to_csv('./data/qrep.csv')

numeric_queries = pd.DataFrame(queryset, columns=numeric_features)
categorical_queries = pd.DataFrame(queryset, columns=categorical_features)

def drop_features(df):
    df.drop('duration', axis=1, inplace=True)
    df.drop('poutcome', axis=1, inplace=True)
    df.drop('default', axis=1, inplace=True)
    return df


numeric_qrt1 = numeric_data.quantile(0.25)
numeric_qrt3 = numeric_data.quantile(0.75)
numeric_std = numeric_data.std()


def impute_numeric(df, column):
    mean = df[column].mean()
    numeric_iqr = numeric_qrt3[column] - numeric_qrt1[column]
    min_val = numeric_qrt1[column] - (1.5*numeric_iqr)
    max_val = numeric_qrt3[column] + (1.5*numeric_iqr)
    col_values = df[column].values
    df[column] = np.where(np.logical_or(col_values < min_val, col_values > max_val), mean, col_values)
    return df


def impute_categorical(df, column):
    df[column] = np.where(df[column].values == 'unknown', df[column].mode(), df[column].values)
    return df


for col in categorical_features:
    trainingset = impute_categorical(trainingset, col)
    queryset = impute_categorical(queryset, col)

for col in numeric_features:
    trainingset = impute_numeric(trainingset, col)
    queryset = impute_numeric(queryset, col)


trainingset = drop_features(trainingset)
queryset = drop_features(queryset)

# ------ Undersampling ----------
sampling_data = trainingset
typeA_count, typeB_count = sampling_data['y'].value_counts()

typeA = trainingset[trainingset['y'] == 'TypeA']
typeB = trainingset[trainingset['y'] == 'TypeB']

typeA_under = typeA.sample(typeB_count)
data_under = pd.concat([typeA_under, typeB], axis=0)

data_X = data_under[data_under.columns[:-1]]
data_X = pd.get_dummies(data_X)
data_y = data_under['y']

queries_x = pd.get_dummies(queryset[queryset.columns[:-1]])

classifiers = []
classifiers.append(('DecisionTree ', DecisionTreeClassifier()))
classifiers.append(('NaiveBayes   ', GaussianNB()))
classifiers.append(('KNN          ', KNeighborsClassifier()))
classifiers.append(('SVM          ', SVC(gamma='auto')))

X, X_test, y, y_test = model_selection.train_test_split(data_X, data_y, test_size=0.25, random_state=13)

def training_classifiers(clf_array):
    for name, classifier in clf_array:
        cls = classifier
        cls = cls.fit(X, y)
        y_pred = cls.predict(X_test)
        accuracy = metrics.accuracy_score(y_test, y_pred)
        recall = metrics.recall_score(y_test, y_pred, average=None)
        hmean = 1.0/((1.0/2.0)*(((1.0/recall[0])) + ((1.0/recall[1]))))
        cm = confusion_matrix(y_test, y_pred)
        print(name, "| Accuracy:", accuracy, "Harmonic Mean:", hmean)


training_classifiers(classifiers)

cls = classifiers[0][1]
y_pred = cls.predict(queries_x)

prediction_data = pd.DataFrame(data=y_pred, index=queries_index)
prediction_data.columns=['pred']
count = prediction_data.groupby('pred').size()
percent = count/len(prediction_data)*100
print(percent)
prediction_data['pred'] = '"' + prediction_data['pred'] + '"'
prediction_data.to_csv('data/predictions.txt', sep=',',  quoting=csv.QUOTE_NONE)
