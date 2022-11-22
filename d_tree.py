from sklearn import tree
import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split
import graphviz
import seaborn
from matplotlib import pyplot as plt

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


# for i in range(1,11):
clf=tree.DecisionTreeClassifier(criterion='gini').fit(X_train,y_train)
dot_data = tree.export_graphviz(clf, 
                filled=True, 
                feature_names=list(X_train),
                class_names=['No rain','rain'],
                special_characters=True)
graph = graphviz.Source(dot_data)
# graph.format = 'png'
# graph.render('output-graph.gv', view=True)
clf.predict(X_test)
# print(clf.score(X_test,y_test))

clf.predict(input)
print(clf.score(input,ans))