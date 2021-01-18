# -*- coding: utf-8 -*-
"""
Created on Sun Mar 24 20:59:36 2019

@author: admin
"""

import pandas as pd
import numpy as np
data=pd.read_excel('XYZCorp_LendingData_Final.xlsx',header=0)
#%%
##Deleted date variables as they were not able to convert to numeric dats y
data.drop('earliest_cr_line', axis=1,inplace=True)
data.drop('last_pymnt_d', axis=1,inplace=True)
data.drop('next_pymnt_d', axis=1,inplace=True)
data.drop('last_credit_pull_d', axis=1,inplace=True)

#%%

print(data)
data.head()
pd.set_option('display.max_columns',None)
print(data)
data.shape
data.isnull().sum()
data.total_rec_prncp.isnull().sum()##Missing value cuming zero..As this variables has been hidden so seperately found mv



#%%
#data_df_rev=pd.DataFrame.copy(data)##always fr all dataets create copy as orignl file gtes corupted saed by nikita ///create copy as data frm server doesnt gets load 
#data_df_rev.dtypes  ####GIVES DATA TYPES
#data_df_rev.describe(include='all')

#%%

data.dtypes  ####GIVES DATA TYPES
#Replacing numeric missing values with mean..Those values whose mv caught is less replaced by mean

data['emp_length Updated'].fillna((data['emp_length Updated'].mean()), inplace=True)
data['revol_util'].fillna((data['revol_util'].mean()), inplace=True)
data['collections_12_mths_ex_med'].fillna((data['collections_12_mths_ex_med'].mean()), inplace=True)
data['tot_coll_amt'].fillna((data['tot_coll_amt'].mean()), inplace=True)
data['tot_cur_bal'].fillna((data['tot_cur_bal'].mean()), inplace=True)
data['total_rev_hi_lim'].fillna((data['total_rev_hi_lim'].mean()), inplace=True)
data.isnull().sum()

#%%
##Here whole data goes in missing values
#Replacing blank values with nan first then fillling it with zero as replacing direct zero nt possinle.
data.inq_last_12m.replace('', np.NaN, inplace=True)
data['inq_last_12m'].fillna(0,inplace=True)

data.total_cu_tl.replace('', np.NaN, inplace=True)
data['total_cu_tl'].fillna(0,inplace=True)

data.inq_fi.replace('', np.NaN, inplace=True)
data['inq_fi'].fillna(0,inplace=True)

data.all_util.replace('', np.NaN, inplace=True)
data['all_util'].fillna(0,inplace=True)

data.max_bal_bc.replace('', np.NaN, inplace=True)
data['max_bal_bc'].fillna(0,inplace=True)

data.open_rv_24m.replace('', np.NaN, inplace=True)
data['open_rv_24m'].fillna(0,inplace=True)

data.open_rv_12m.replace('', np.NaN, inplace=True)
data['open_rv_12m'].fillna(0,inplace=True)

data.il_util.replace('', np.NaN, inplace=True)
data['il_util'].fillna(0,inplace=True)

data.total_bal_il.replace('', np.NaN, inplace=True)
data['total_bal_il'].fillna(0,inplace=True)

data.mths_since_rcnt_il.replace('', np.NaN, inplace=True)
data['mths_since_rcnt_il'].fillna(0,inplace=True)

data.open_il_24m.replace('', np.NaN, inplace=True)
data['open_il_24m'].fillna(0,inplace=True)

data.open_il_12m.replace('', np.NaN, inplace=True)
data['open_il_12m'].fillna(0,inplace=True)

data.open_il_6m.replace('', np.NaN, inplace=True)
data['open_il_6m'].fillna(0,inplace=True)

data.open_acc_6m.replace('', np.NaN, inplace=True)
data['open_acc_6m'].fillna(0,inplace=True)

data.mths_since_last_major_derog.replace('', np.NaN, inplace=True)
data['mths_since_last_major_derog'].fillna(0,inplace=True)

data.mths_since_last_record.replace('', np.NaN, inplace=True)
data['mths_since_last_record'].fillna(0,inplace=True)

data.mths_since_last_delinq.replace('', np.NaN, inplace=True)
data['mths_since_last_delinq'].fillna(0,inplace=True)

data.isnull().sum()
data.shape


#%%
colname=[ 'term', 'grade','sub_grade', 'home_ownership Updated', 'verification_status Updated','pymnt_plan',
'initial_list_status', 'application_type']###all here are list names of only cat variable as fr label encodung cat data is needed
colname


####For preprocessing the data
###below linefrom sklearn till fit transfrm code will be same always fr all datsets  in python compulary nly changing nly datset name
###convertg cat data to num data
from sklearn import preprocessing    #####sklearn is used to run label encoder
le={}      ##le=label encoding,{} caleed as empty dictionary//## creating dictionery

for x in colname:
    le[x]=preprocessing.LabelEncoder()
    
for x in colname:
    data[x]=le[x].fit_transform(data[x]) ###### transforming the data into dictionery////fit transform used to pass orignal cat data in indivdual varible n cnvert it into numberss.. //fit transform used to pass orignal traing set of data to model
    
data.head()
data.dtypes  
#%%
###Splitting data into test train

split_date ='2015-05-31'
X = data.loc[data['issue_d'] <= split_date]
Y = data.loc[data['issue_d'] > split_date]
data.drop('issue_d', axis=1,inplace=True)
data.shape  ####(855969 rows and 56 columns)

X=data.values[:,:-1] 
Y=data.values[:,-1]
#%%
#Scaling
from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()##scaling allows us to convert all independent values //to bring data in uniform way we use scaling 
scaler.fit(X) ##fit func pass orignal data to scaler object n find out new rang of values
X=scaler.transform(X)
print(X)##o/p lies btwn range -3 t0 3 ....Doubttttttttttttttttttt all valuesss are coming in negativeeeeeeeeeeeee
Y=Y.astype(int)
print(Y)

#%%
#Model Building
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test=train_test_split(X,Y,test_size=0.3,random_state=10)##c notes 20 jan 3page,testsize=0.3 indctes what ratio mgt be used to split data, random state=10 is an argument it cud be same numerc value that we can apss as an argument//random state=10 value can be any be 11,23 12 ne no but always use 10 as nikita use 10 fr all datsetes in python
## if we do not provide random_state it wil take random observationes & ouput 
##will b different  for evry execution
print(X_train.shape)
print(X_test.shape)


#%%
####Logistic Regression

from sklearn.linear_model import LogisticRegression
classifier=LogisticRegression() ###round brackets() are compulsary//classifer always point to LOg reg.Classifer usd fr any of algorthm used fr classifctn purpose such as DT,log reg
##fitting trainoing data to the model
classifier.fit(X_train,Y_train)##fit function used to training uh model
##prediction on test data
Y_pred=classifier.predict(X_test)
print(list(zip(Y_test,Y_pred)))############doubt i guess wrong 
#%%
###commonly usd steps always to find accuracy
###Vimpppppppppppppp  fr details c nts 20jan 5,6,7,8 page fr below details
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm=confusion_matrix(Y_test,Y_pred)
print(cfm) ### c nts 20jan 5page n 6th page
print("Classification report:")
print(classification_report(Y_test,Y_pred))### c nts 20jan 5page n 6th page
acc=accuracy_score(Y_test,Y_pred)
print("Accuracy of the model:",acc) ###acc=99.73






#%%
##store the predicted probabilities
###c notes 9th page
##c description in 9feb small diary nts fr relation btwn line print(list(zip(Y_test,Y_pred))) with print(y_pred_prob)
y_pred_prob=classifier.predict_proba(X_test)
print(y_pred_prob)##o/p 1st obs has 0.96 prob to belong to class 0 n 0.030 is proba belong to class 1..compare line 148 n line 129..0.22 0class 0.77 greater so in line 117(0,1) bcz 0.77 greater than 0.22 in line 132
###c small diary 9 feb adult dataset logistic regression



y_pred_class=[]
for value in y_pred_prob[:,1]:   ###[:,1] indctes : whole rows including 1st col
    if value > 0.6:
        y_pred_class.append(1)
    else:
            y_pred_class.append(0)
print(y_pred_class)




from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm=confusion_matrix(Y_test,y_pred_class)
print(cfm) ### c nts 20jan 5page n 6th page
print("Classification report:")
print(classification_report(Y_test,y_pred_class))### c nts 20jan 5page n 6th page
acc=accuracy_score(Y_test,y_pred_class)
print("Accuracy of the model:",acc)

#%%
#Univariate feature selection method

from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2


test = SelectKBest(score_func=chi2, k=20)##test is object k=11 how many variables i want to select
fit1 = test.fit(X, Y)##fit to find chi sqaure value
print(X)

print(fit1.scores_)#### it wil retuen all 11 chi2 values
print(list(zip(colname,fit1.get_support())))##get supprt funct which return boolean list if valyes
X = fit1.transform(X)##fit1 selected 11 variables total 14 variabless.
## transform(X)=original 14 elements wala x, 
##it wil subset with 11 elements and will be stored i X
print(X)


from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
cfm=confusion_matrix(Y_test,y_pred_class)
print(cfm) ### c nts 20jan 5page n 6th page
print("Classification report:")
print(classification_report(Y_test,y_pred_class))### c nts 20jan 5page n 6th page
acc=accuracy_score(Y_test,y_pred_class)
print("Accuracy of the model:",acc)











#%%
##Dropping var fr tpppp

#data.drop('inq_last_12m', axis=1,inplace=True)
#data.drop('total_cu_tl', axis=1,inplace=True)
#data.drop('inq_fi', axis=1,inplace=True)

