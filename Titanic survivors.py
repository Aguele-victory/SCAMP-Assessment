#importing libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#accessing the datasets
train_set = pd.read_csv('train.csv')
test_set = pd.read_csv('test.csv')

#Duplicating the datasets for later use.
train_set1 = train_set.copy()
test_set1 = test_set.copy()

#data exploration
#checking the first ten enties of both data
train_set.head(10)
test_set.head(10) 
#checking the information contained in each to know if we have missing values
train_set.info()
test_set.info()
train_set.describe()
test_set.describe()
#checking for the titles of each column
train_set.keys()
test_set.keys()
"""from our exploration we can see that first train_set has more columns than our test_set, it contains a column 
indicating survivors. Also there are missing values in both data set especially in columns; age,cabin, fare and 
embarked. as we would not be needing some of the columns, it can either be deleted or left out when indexing.two 
columns 'SibSp' and 'Parch' would be merged together as they show size of family for each passenger."""

#DATA PREPROCESSING
#dropping columns not needed
train_set = train_set.drop(['PassengerId','Name','Cabin','Ticket'], axis=1)
test_set = test_set.drop(['PassengerId','Name','Cabin','Ticket'], axis=1)

#dealing with missing values
#AGE
train_set['Age'] = train_set['Age'].fillna(train_set['Age'].mean())
test_set['Age'] = test_set['Age'].fillna(test_set['Age'].mean())
#Fare
train_set['Fare'] = train_set['Fare'].fillna(train_set['Fare'].mean())
test_set['Fare'] = test_set['Fare'].fillna(test_set['Fare'].mean())
#Embarked
""" from the info, embarked has two missing values which occurs only in the train_set, so we'll index out the rows 
with missing values and delete it from the dataset"""
miss_emb = pd.isnull(train_set['Embarked'])
train_set[miss_emb]
#so index 61 and 829 are the rows with missing values which we'll be deleting
train_set = train_set.drop([61,829], axis=0)

#merging columns SibSp and Parch since it shows family size
train_set['size_family'] = train_set['Parch'] + train_set['SibSp']
test_set['size_family'] = test_set['Parch'] + test_set['SibSp']
#dropping the two columns
train_set = train_set.drop(['Parch', 'SibSp'], axis=1)
test_set = test_set.drop(['Parch', 'SibSp'], axis=1)

#Splitting into dependent and independent variables
X_train = train_set.iloc[:, 1:].values
Y_train = train_set.iloc[:, 0:1].values
X_test = test_set.iloc[:, 0:].values

#ENCODING CATEGORICAL DATA
#X_train
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer( transformers=[('one_hot_encoder',OneHotEncoder(categories='auto'),[1,4])],remainder='passthrough')
X_train = np.array(ct.fit_transform(X_train),dtype=np.float)
#X_test
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
ct = ColumnTransformer( transformers=[('one_hot_encoder',OneHotEncoder(categories='auto'),[1,4])],remainder='passthrough')
X_test = np.array(ct.fit_transform(X_test),dtype=np.float)

# Feature Scaling
#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#X_train = sc_X.fit_transform(X_train)
#X_test = sc_X.transform(X_test)

#fitting decision tree to training set
from sklearn.tree import DecisionTreeClassifier
classifier = DecisionTreeClassifier(criterion='entropy', random_state=0)
classifier.fit(X_train,Y_train)

#Prediction using test results
Y_pred = classifier.predict(X_test)

accuracy_score = round(classifier.score(X_train, Y_train) * 100, 2)
print(accuracy_score)

#kaggle submission
New = pd.DataFrame({'PassengerId': test_set1['PassengerId'],'Survived': Y_pred})
#converting to csv file
New.to_csv('Titanic Prediction.csv')


