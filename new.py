import json
import numpy as np
import pandas as pd
import requests
import matplotlib.pyplot as plt

### read the data
loan_2007= pd.read_csv('/Users/cuiheng/Downloads/LoanStats3a.csv', skiprows=1)
loan_2012= pd.read_csv('/Users/cuiheng/Downloads/LoanStats3b.csv', skiprows=1)
loan_2014= pd.read_csv('/Users/cuiheng/Downloads/LoanStats3c.csv', skiprows=1)
all_feature_2007 = loan_2007.columns.values.tolist()
all_feature_2012 = loan_2012.columns.values.tolist()
all_feature_2014 = loan_2014.columns.values.tolist()

loan_combined=pd.concat([loan_2007,loan_2012,loan_2014])

### data_visualization and cleaning
### drop the categroical features which have too many categories or only one category
### drop the features which have 80% missing values
drop_feature=[]
for i in loan_combined.columns:
    if (len(pd.unique(loan_combined[i].value_counts())) ==1 or len(pd.unique(loan_combined[i].value_counts())) >500 ) and loan_combined[i].dtypes=='object':
        drop_feature.append(i)
    if loan_combined[i].isnull().mean()>.8:
        drop_feature.append(i)
loan_combined_1 = loan_combined.drop(drop_feature,1)


###

### here, we only care about the loans with 36 months
loan_combined_2 = loan_combined_1[loan_combined_1['term']==' 36 months']  ###337996 loans
loan_combined_3 = loan_combined_2.drop('term', 1)
loan_combined_3['interest_rate'] = loan_combined_3['int_rate'].apply(lambda x: float(x.replace('%',''))/100)
drop_irr = ['int_rate' ,'emp_title', 'issue_d','pymnt_plan','desc', 'title','last_pymnt_d','last_credit_pull_d','policy_code','hardship_flag','earliest_cr_line']
loan_combined_4 = loan_combined_3.drop(drop_irr,1)
all_feature = loan_combined_4.columns
obj_feature = loan_combined_4.select_dtypes(include=['object']).columns
num_feature = all_feature.difference(obj_feature)


def model_matrix(loan, columns):
    dummified_cols = pd.get_dummies(loan[columns])
    loan = loan.drop(columns, axis = 1, inplace=False)
    loan_new = pd.concat([loan,dummified_cols],1)
    return loan_new


y= loan_combined_4['loan_status']
x = loan_combined_4.drop("loan_status", axis=1, inplace = False)
x= model_matrix(x, obj_feature.drop('loan_status'))


from sklearn.preprocessing import MinMaxScaler
Scaler = MinMaxScaler()
x= x.fillna(value = 0)
x[num_feature] = Scaler.fit_transform(x[num_feature])


import sklearn.model_selection as sk
x_train, x_test, y_train, y_test = sk.train_test_split(x, y, random_state=2017)

x_train_1 = x_train[:10000]
y_train_1 = y_train[:10000]

from sklearn.neighbors import KNeighborsClassifier
knn=KNeighborsClassifier(n_neighbors = 5, metric='euclidean')
knn.fit(x_train_1,y_train_1)
knn.score(x_test,y_test)


aaaaaaaa