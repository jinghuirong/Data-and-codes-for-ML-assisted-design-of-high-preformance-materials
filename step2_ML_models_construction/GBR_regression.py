from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn import ensemble
from sklearn.model_selection import KFold, GridSearchCV
from pylab import *

df=pd.read_csv('generated_features.csv')
X=df.drop(['target','name'],axis=1)
Y=df['target']
column=list(X)
scaler = StandardScaler()
X_scaled=pd.DataFrame(scaler.fit_transform(X))
X_scaled.columns=X.columns

Xtrain,Xtest,Ytrain,Ytest=train_test_split(X_scaled, Y,random_state=0,test_size=0.2)

param_grid=[{'max_depth':range(2,7,1),'learning_rate':[0.03,0.04,0.05,0.06,0.07,0.08
            ,0.09,0.1,0.11,0.12,0.13]
            ,'min_samples_leaf':range(2,6)}]
GBR=ensemble.GradientBoostingRegressor(random_state=42, n_estimators=200)
kf=KFold(n_splits=5,shuffle=True,random_state=0)
grid_search=GridSearchCV(GBR,param_grid,cv=kf,scoring='neg_mean_absolute_error')
grid_search.fit(Xtrain,Ytrain)
grid_search.best_estimator_

final_model=grid_search.best_estimator_
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score
print(mean(abs(cross_val_score(final_model, Xtrain,Ytrain, scoring='neg_mean_absolute_error', cv=kf))))
print(mean(abs(cross_val_score(final_model, Xtrain,Ytrain, scoring='r2', cv=kf))))
y_test_predict=final_model.predict(Xtest)
y_train_predict=final_model.predict(Xtrain)

from sklearn.metrics import r2_score, mean_absolute_error
r2train=[]
r2test=[]
MAEtrain=[]
MAEtest=[]
for i in range (0,50):
    Xtrain,Xtest,Ytrain,Ytest=train_test_split(X_scaled, Y,random_state=i,test_size=0.20)
    GBR=final_model
    GBR.fit(Xtrain,Ytrain)
    y_train_predict=GBR.predict(Xtrain)
    y_test_predict=GBR.predict(Xtest)
    r2_train=r2_score(Ytrain,y_train_predict)
    r2_test=r2_score(Ytest,y_test_predict)
    r2train.append(r2_train)
    r2test.append(r2_test)
    kf=KFold(n_splits=5,shuffle=True,random_state=0)
    MAE_train=mean(abs(cross_val_score(final_model, Xtrain,Ytrain, scoring='neg_mean_absolute_error', cv=kf)))
    MAE_test=mean_absolute_error(Ytest,y_test_predict)
    MAEtrain.append(MAE_train)
    MAEtest.append(MAE_test)
    
print(np.std(MAEtrain))
print(np.std(MAEtest))
print(f"The MAE of train set is {mean(MAEtrain)},The R2 of train set is {mean(r2train)}")
print(f"The MAE of test set is {mean(MAEtest)}, The R2 of test set is {mean(r2test)}")
print(f"The MAE of test set is {min(MAEtest)}, The R2 of test set is {max(r2test)}")
print(f"The MAE of test set is {max(MAEtest)}, The R2 of test set is {min(r2test)}")