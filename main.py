import numpy as np
import pandas as pd

data = pd.read_csv("train.csv")

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
                angle+=np.pi
            if i == "N":
                angle+=np.pi/2
            if i == "S":
                angle+=np.pi*3/2
        angle=angle/(len(w))
    except:
        pass
    return angle


#將資料分成訓練組及測試組
from sklearn.model_selection import train_test_split

data[["Attribute8","Attribute10"]]=data[["Attribute8","Attribute10"]].applymap(convert_wind)

data[["Attribute16","Attribute17"]]=data[["Attribute16","Attribute17"]].applymap(lambda x: 0 if x=="No" else 1)

data=data.fillna(0)

X = data[["Attribute2","Attribute3","Attribute4","Attribute5","Attribute6","Attribute7","Attribute8","Attribute9","Attribute10","Attribute11","Attribute12","Attribute13","Attribute14","Attribute15","Attribute16"]]
y = data['Attribute17']


X_train, X_test, y_train, y_test = train_test_split(X,y,random_state=)

from sklearn.svm import SVC

#載入GridSearchCV
from sklearn.model_selection import GridSearchCV

#GridSearchCV是建立一個dictionary來組合要測試的參數
param_grid = {'C':[1,10,1000],'gamma':[1,0.01,0.0001]}
# param_grid = {'C':[10],'gamma':[0.0001]}

grid = GridSearchCV(SVC(),param_grid,n_jobs=5,verbose=3)

#利用剛剛設定的參數來找到最適合的模型
grid.fit(X_train,y_train)

print(grid.best_estimator_)
# grid.best_estimator_=SVC(C=1, gamma=0.001).fit(X_train,y_train)

test_data=pd.read_csv("test.csv")
test_data[["Attribute8","Attribute10"]]=test_data[["Attribute8","Attribute10"]].applymap(convert_wind)
test_data[["Attribute16"]]=test_data[["Attribute16"]].applymap(lambda x: 0 if x=="No" else 1)

#利用剛剛的最佳參考再重新預測測試組
grid_predictions = grid.predict(X_test)

from sklearn.metrics import classification_report, confusion_matrix
#評估新參考的預測結果好壞
print('confusion_matrix : ')
print(confusion_matrix(y_test,grid_predictions))
print('classification_report : ')
print(classification_report(y_test,grid_predictions))

#利用剛剛的最佳參考再重新預測測試組
grid_predictions = grid.predict(test_data[["Attribute2","Attribute3","Attribute4","Attribute5","Attribute6","Attribute7","Attribute8","Attribute9","Attribute10","Attribute11","Attribute12","Attribute13","Attribute14","Attribute15","Attribute16"]])

ans=list(pd.read_csv("ex_submit.csv")["ans"])

from sklearn.metrics import classification_report, confusion_matrix
#評估新參考的預測結果好壞
print('confusion_matrix : ')
print(confusion_matrix(ans,grid_predictions))
print('classification_report : ')
print(classification_report(ans,grid_predictions))