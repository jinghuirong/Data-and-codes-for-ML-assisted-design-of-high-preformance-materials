from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split
import pandas as pd
import seaborn as sns
from pylab import *
import os
from sklearn import ensemble
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.model_selection import cross_val_score

def R2_MAE(m):
    files=[i for i in os.listdir('../new_training_data') if i.startswith(f'f{m}')]
    files.sort()
    (R10,MAE10)=([],[])
    for file in files:
        dfd=pd.read_csv(file)
        dff=pd.read_csv('generated_features.csv')
        df=pd.concat([dfd, dff], axis = 0)
        X=df.drop(['target','formula','name'],axis=1)
        Y=df['target']
        column=list(X)
        df1 = pd.read_csv('testset_features.csv')
        Xt=df1.drop(['target','formula','name'],axis=1)
        Yt=df1['target']
        scaler = StandardScaler()
        X_scaled=pd.DataFrame(scaler.fit_transform(X))
        Xt_scaled=pd.DataFrame(scaler.transform(Xt))
        X_scaled.columns=X.columns
        Xt_scaled.columns=Xt.columns
        param_grid=[{'max_depth':range(2,7,1),'learning_rate':[0.03,0.04,0.05,0.06,0.07,0.08
                    ,0.09,0.1,0.11,0.12,0.13],'min_samples_leaf':range(2,6)}]
        GBR=ensemble.GradientBoostingRegressor(random_state=42, n_estimators=200)
        kf=KFold(n_splits=5,shuffle=True,random_state=0)
        grid_search=GridSearchCV(GBR,param_grid,cv=kf,scoring='neg_mean_absolute_error')
        grid_search.fit(X_scaled,Y)
        grid_search.best_estimator_
        final_model=grid_search.best_estimator_
        y_test_predict=final_model.predict(Xt_scaled)
        R10.append(r2_score(Yt,y_test_predict))
        MAE10.append(mean_absolute_error(Yt,y_test_predict))
    print(R10)
    print(MAE10)
    print(mean(R10))
    print(mean(MAE10))
    