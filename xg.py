from xgboost import XGBClassifier,XGBRegressor
import pandas as pd
import numpy as np 
import sklearn
from sklearn.model_selection import train_test_split
import datetime
from itertools import product
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import GroupKFold
from sklearn import metrics
# form sklearn import me
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA

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

def mean_norm(df_input):
    # noramlization-----------------------------------------------------------------------------------------
    return df_input.apply(lambda x: ((x-x.mean())/ x.std()), axis=0)
    # return df_input.apply(lambda x: (x/ (x.max()-x.min())), axis=0)

params = { 'max_depth': [3,6],
           'learning_rate': [ 0.05, 0.1],
           'n_estimators': [100, 500],
           'colsample_bytree': [0.3, 0.7]}

compoment=13

#================================================train data processing=========================================           

data = pd.read_csv("train.csv")
data["Attribute1"] = pd.to_datetime(data["Attribute1"])
data["temp_diff"] = data["Attribute4"]-data["Attribute3"]

# first_date=datetime.datetime.strptime(data["Attribute1"][0],"%Y-%m-%d")

data["Attribute1"]=data["Attribute1"].apply(lambda r: conver_date(r,data["Attribute1"].min()))

data[["Attribute8","Attribute10"]]=data[["Attribute8","Attribute10"]].applymap(convert_wind)

data[["Attribute16","Attribute17"]]=data[["Attribute16","Attribute17"]].applymap(lambda x: 0 if x=="No" else 1)

N=data.isnull().sum()

# print(N)
print()
data = data.dropna()
# print(data.shape)

min_max=MinMaxScaler()
data[["Attribute3","Attribute4","Attribute5","Attribute6","Attribute7","Attribute9",
"Attribute11","Attribute12","Attribute13","Attribute14","Attribute15","temp_diff"]] = mean_norm(data[["Attribute3","Attribute4","Attribute5","Attribute6","Attribute7","Attribute9",
"Attribute11","Attribute12","Attribute13","Attribute14","Attribute15","temp_diff"]])
# data["Attribute3"]=pd.DataFrame(min_max.fit_transform(data,columns))
# data["Attribute4"]=pd.DataFrame(min_max.fit_transform(data["Attribute4"]))
# data["Attribute5"]=pd.DataFrame(min_max.fit_transform(data["Attribute5"]))
# data["Attribute6"]=pd.DataFrame(min_max.fit_transform(data["Attribute6"]))
# data["Attribute7"]=pd.DataFrame(min_max.fit_transform(data["Attribute7"]))
# data["Attribute9"]=pd.DataFrame(min_max.fit_transform(data["Attribute9"]))
# data["Attribute11"]=pd.DataFrame(min_max.fit_transform(data["Attribute11"]))
# data["Attribute12"]=pd.DataFrame(min_max.fit_transform(data["Attribute12"]))
# data["Attribute13"]=pd.DataFrame(min_max.fit_transform(data["Attribute13"]))
# data["Attribute14"]=pd.DataFrame(min_max.fit_transform(data["Attribute14"]))
# data["Attribute15"]=pd.DataFrame(min_max.fit_transform(data["Attribute15"]))
# data["temp_diff"]=pd.DataFrame(min_max.fit_transform(data["temp_diff"]))


#===============================================data balancing==============================================
X=data

# for i in data.columns:    
#     X[i]=X[i].fillna(X[i].mean())

rain_data=X[X["Attribute17"]==1]
no_rain_data=X[X["Attribute17"]==0]
# print(rain_data.shape)
# print(no_rain_data.shape)

no_rain_rand=no_rain_data.sample(int(rain_data.shape[0]*3))

# print(no_rain_rand.shape)


X=pd.concat([no_rain_rand,rain_data])

y = X['Attribute17'] 
# # X = data.drop(["Attribute17","Attribute6","Attribute13","Attribute7","Attribute14","Attribute1","Attribute9","Attribute16"],axis=0)
X = X.drop(columns=["Attribute1","Attribute2", "Attribute17", "Attribute8", "Attribute10"])
pca = PCA(n_components=compoment)
X_reduced = pca.fit_transform(X)
# print("rain: ",y[y==1].shape[0])
# print("no rain: ",y[y==0].shape[0])

# ===============================================test data processing==============================================
test_data=pd.read_csv("test.csv")
t=test_data.isnull().sum()
# print(t)

test_data["temp_diff"] = test_data["Attribute4"]-test_data["Attribute3"]
test_data["Attribute1"]=pd.to_datetime(test_data["Attribute1"])
test_data["Attribute1"]=test_data["Attribute1"].apply(lambda r: conver_date(r,test_data["Attribute1"].min()))
test_data[["Attribute8","Attribute10"]]=test_data[["Attribute8","Attribute10"]].applymap(convert_wind)
test_data[["Attribute16"]]=test_data[["Attribute16"]].applymap(lambda x: 0 if x=="No" else 1)

test_data[["Attribute3","Attribute4","Attribute5","Attribute6","Attribute7","Attribute9",
"Attribute11","Attribute12","Attribute13","Attribute14","Attribute15","temp_diff"]] = mean_norm(test_data[["Attribute3","Attribute4","Attribute5","Attribute6","Attribute7","Attribute9",
"Attribute11","Attribute12","Attribute13","Attribute14","Attribute15","temp_diff"]])
test_data = test_data.drop(columns=["Attribute1","Attribute2", "Attribute8", "Attribute10"])
pca = PCA(n_components=compoment)
test_data_reduced = pca.fit_transform(test_data)
# test_data["Attribute3"]=pd.DataFrame(min_max.fit_transform(test_data["Attribute3"]))
# test_data["Attribute4"]=pd.DataFrame(min_max.fit_transform(test_data["Attribute4"]))
# test_data["Attribute5"]=pd.DataFrame(min_max.fit_transform(test_data["Attribute5"]))
# test_data["Attribute6"]=pd.DataFrame(min_max.fit_transform(test_data["Attribute6"]))
# test_data["Attribute7"]=pd.DataFrame(min_max.fit_transform(test_data["Attribute7"]))
# test_data["Attribute9"]=pd.DataFrame(min_max.fit_transform(test_data["Attribute9"]))
# test_data["Attribute11"]=pd.DataFrame(min_max.fit_transform(test_data["Attribute11"]))
# test_data["Attribute12"]=pd.DataFrame(min_max.fit_transform(test_data["Attribute12"]))
# test_data["Attribute13"]=pd.DataFrame(min_max.fit_transform(test_data["Attribute13"]))
# test_data["Attribute14"]=pd.DataFrame(min_max.fit_transform(test_data["Attribute14"]))
# test_data["Attribute15"]=pd.DataFrame(min_max.fit_transform(test_data["Attribute15"]))
# test_data["temp_diff"]=pd.DataFrame(min_max.fit_transform(test_data["temp_diff"]))
a=0

# ====================================================get ans=======================================================
ans=list(pd.read_csv("ex_submit.csv")["ans"])

# ==================================================split train data and test data==========================================================
X_train, X_test, y_train, y_test = train_test_split(X_reduced,y,train_size=0.7, random_state=0, shuffle=True)

# print(X_train)
# exit(0)

from sklearn.tree import DecisionTreeClassifier,plot_tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from matplotlib import pyplot as plt

for i in range(10,110,10):

    # ------------------------------------------RandomForestClassifier-----------------------------------------
    RandomForest=RandomForestClassifier(random_state=0,max_depth=int(i/10))
    # print(sklearn.metrics.get_scorer_names())
    RandomForest.fit(X_train, y_train)
    # 使用訓練資料預測分類
    # predicted = xgboostModel.predict(X_train)
    inputs=test_data_reduced
    predict_ans=RandomForest.predict(inputs)

    # 預測成功的比例
    print("RandomForestClassifier:")
    print('n_estimators=',i,' Train: ',RandomForest.score(X_train,y_train))
    print('n_estimators=',i,' Test: ',RandomForest.score(X_test,y_test))
    print('n_estimators=',i,' Real test: ',metrics.accuracy_score(predict_ans,ans))
    print()

    # ------------------------------------DecisionTreeClassifier------------------------------------------------
    DecisionTree=DecisionTreeClassifier(random_state=0,max_depth=i)
    # print(sklearn.metrics.get_scorer_names())
    DecisionTree.fit(X_train, y_train)
    # 使用訓練資料預測分類
    # predicted = xgboostModel.predict(X_train)
    inputs=test_data_reduced
    predict_ans=DecisionTree.predict(inputs)

    # 預測成功的比例
    print("DecisionTreeClassifier:")
    print('max_depth=',i,' Train: ',DecisionTree.score(X_train,y_train))
    print('max_depth=',i,' Test: ',DecisionTree.score(X_test,y_test))
    print('max_depth=',i,' Real test: ',metrics.accuracy_score(predict_ans,ans))
    print()

    # --------------------------------------XGBClassifier-----------------------------------------------------------
    xg = XGBClassifier(tree_method='gpu_hist', gpu_id=0,subsample=i/100)

    xg.fit(X_train, y_train)
    # 使用訓練資料預測分類
    # predicted = xgboostModel.predict(X_train)
    inputs=test_data_reduced
    predict_ans=xg.predict(inputs)

    # 預測成功的比例
    print("XGBClassifier:")
    print('subsample=',i/100,' Train: ',xg.score(X_train,y_train))
    print('subsample=',i/100,' Test: ',xg.score(X_test,y_test))
    print('subsample=',i/100,' Real test: ',metrics.accuracy_score(predict_ans,ans))
    print()

    # --------------------------------------KNN-----------------------------------------------------------
    knn = KNeighborsClassifier(n_neighbors=i)

    knn.fit(X_train, y_train)
    # 使用訓練資料預測分類
    # predicted = xgboostModel.predict(X_train)
    inputs=test_data_reduced
    predict_ans=knn.predict(inputs)

    # 預測成功的比例
    print("KNN:")
    print('subsample=',i/100,' Train: ',knn.score(X_train,y_train))
    print('subsample=',i/100,' Test: ',knn.score(X_test,y_test))
    print('subsample=',i/100,' Real test: ',metrics.accuracy_score(predict_ans,ans))
    print()


    # xg = XGBClassifier(tree_method='gpu_hist', gpu_id=0,subsample=i/100,eta=0.1 )

    # ------------------------------------plot prediction distribution-----------------------------------------------
    cf_metrix = metrics.confusion_matrix(ans,predict_ans)
    label=[0,1]
    sns.heatmap(cf_metrix,annot=True,xticklabels=label,yticklabels=label,cmap='Blues')
    plt.show(block =False)
    fig=plt.figure(figsize=(15,10))
    # plot_tree(xgboostModel,feature_names=X.columns,filled=True,rounded=True)
    # plt.show()
pass
