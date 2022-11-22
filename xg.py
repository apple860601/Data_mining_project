from xgboost import XGBClassifier,XGBRegressor
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import datetime

def convert_wind(w):
    r=-1
    pos=1
    angle=-1
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
        pass
    return angle

def conver_date(d):
    


data = pd.read_csv("train.csv")

first_date=data["Attribute1"][0]

data["Attribute1"]=data["Attribute1"].applymap(lambda r: conver_date(r,first_date), axis=1)

data[["Attribute8","Attribute10"]]=data[["Attribute8","Attribute10"]].applymap(convert_wind)

data[["Attribute16","Attribute17"]]=data[["Attribute16","Attribute17"]].applymap(lambda x: 0 if x=="No" else 1)

data=data.fillna(0)

X = data[["Attribute1","Attribute2","Attribute3","Attribute4","Attribute5","Attribute6","Attribute7","Attribute8","Attribute9","Attribute10","Attribute11","Attribute12","Attribute13","Attribute14","Attribute15","Attribute16"]]
y = data['Attribute17']

for i in range(1,220,20):
    X_train, X_test, y_train, y_test = train_test_split(X,y,train_size=0.4)

    # 建立 XGBClassifier 模型
    xgboostModel = XGBClassifier(n_estimators=i, learning_rate= 0.4)
    # 使用訓練資料訓練模型
    xgboostModel.fit(X_train, y_train)
    # 使用訓練資料預測分類
    predicted = xgboostModel.predict(X_train)

    # 預測成功的比例
    print('n_estimators=',i,' 訓練集: ',xgboostModel.score(X_train,y_train))
    print('n_estimators=',i,' 測試集: ',xgboostModel.score(X_test,y_test))

    test_data=pd.read_csv("test.csv")
    test_data[["Attribute8","Attribute10"]]=test_data[["Attribute8","Attribute10"]].applymap(convert_wind)

    test_data[["Attribute16"]]=test_data[["Attribute16"]].applymap(lambda x: 0 if x=="No" else 1)

    test_data=test_data.fillna(0)
    a=0
    ans=list(pd.read_csv("ex_submit.csv")["ans"])
    input=test_data[["Attribute1","Attribute2","Attribute3","Attribute4","Attribute5","Attribute6","Attribute7","Attribute8","Attribute9","Attribute10","Attribute11","Attribute12","Attribute13","Attribute14","Attribute15","Attribute16"]]

    print('n_estimators=',i,' 實際測試: ',xgboostModel.score(input,ans))
    print('n_estimators=',i,' 實際測試: ',xgboostModel.predict_proba(input))
