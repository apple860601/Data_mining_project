from xgboost import XGBClassifier,XGBRegressor
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import datetime
from itertools import product
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold

def convert_wind(w):
    r=-1
    pos=1
    angle=0
    # print(w)
    try:
        for i in w:
            if i == "E":
                angle+=0
            if i == "W":
                angle+=180
            if i == "N":
                angle+=90
            if i == "S":
                angle+=270
        angle=angle/(len(w))
    except:
        angle=-1
    return angle

def conver_date(d,first_date):
    # d1=datetime.datetime.strptime(d,"%Y-%m-%d")
    return (d - first_date).days

params = { 'max_depth': [3,6],
           'learning_rate': [ 0.05, 0.1],
           'n_estimators': [100, 500],
           'colsample_bytree': [0.3, 0.7]}

data = pd.read_csv("train.csv")
data["Attribute1"] = pd.to_datetime(data["Attribute1"])

# first_date=datetime.datetime.strptime(data["Attribute1"][0],"%Y-%m-%d")

data["Attribute1"]=data["Attribute1"].apply(lambda r: conver_date(r,data["Attribute1"].min()))

data[["Attribute8","Attribute10"]]=data[["Attribute8","Attribute10"]].applymap(convert_wind)

data[["Attribute16","Attribute17"]]=data[["Attribute16","Attribute17"]].applymap(lambda x: 0 if x=="No" else 1)

data=data.fillna(0)

X = data[["Attribute3","Attribute4","Attribute5","Attribute6","Attribute7","Attribute12","Attribute14","Attribute15","Attribute16"]]
y = data['Attribute17']
X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.7)

# for learning,estimators,max_depth in product(range(0,60,10),range(0,51,5),range(1,11)):
    # 建立 XGBClassifier 模型
    # xgboostModel = XGBClassifier(n_estimators=estimators, learning_rate= learning/100,max_depth=10,tree_method='gpu_hist', gpu_id=0)

xg2 = XGBClassifier(random_state=1,tree_method='gpu_hist', gpu_id=0)
xgboostModel = GridSearchCV(estimator=xg2, 
                param_grid=params,
                scoring='neg_mean_squared_error', 
                verbose=3)
# 使用訓練資料訓練模型
xgboostModel.fit(X_train, y_train)
# 使用訓練資料預測分類
predicted = xgboostModel.predict(X_train)

# 預測成功的比例
# print('n_estimators=',i,' 訓練集: ',xgboostModel.score(X_train,y_train))
# print('n_estimators=',i,' 測試集: ',xgboostModel.score(X_test,y_test))

test_data=pd.read_csv("test.csv")
test_data["Attribute1"]=pd.to_datetime(test_data["Attribute1"])
# print(test_data["Attribute1"].min())
# first_test_date=datetime.datetime.strptime(test_data[pd.to_datetime(test_data["Attribute1"])],"%Y-%m-%d")
test_data["Attribute1"]=test_data["Attribute1"].apply(lambda r: conver_date(r,test_data["Attribute1"].min()))
test_data[["Attribute8","Attribute10"]]=test_data[["Attribute8","Attribute10"]].applymap(convert_wind)

test_data[["Attribute16"]]=test_data[["Attribute16"]].applymap(lambda x: 0 if x=="No" else 1)

test_data=test_data.fillna(0)
a=0
ans=list(pd.read_csv("ex_submit.csv")["ans"])
input=test_data[["Attribute3","Attribute4","Attribute5","Attribute6","Attribute7","Attribute12","Attribute14","Attribute15","Attribute16"]]

print('n_estimators=',' 實際測試: ',xgboostModel.score(input,ans))
print()
    # print('n_estimators=',i,' 實際測試: ',xgboostModel.predict_proba(input))
