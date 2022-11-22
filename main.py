import numpy as np
import pandas as pd
from sklearn.svm import SVC
from itertools import product
import multiprocessing as mp

data = pd.read_csv("train.csv")
score_list=[]

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


def train_module(x,y,C,gamma,return_dict,procnum):
    X_train, X_test, y_train, y_test = train_test_split(x,y,train_size=0.4)

    grid=SVC(C=C,gamma=gamma )

    #利用剛剛設定的參數來找到最適合的模型
    grid.fit(X_train,y_train)

    print(grid.score(X_test,y_test))
    # score_list.append(grid.score(X_test,y_test))
    return_dict[procnum]={'score':grid.score(X_test,y_test),'grid':grid}
    # return_dict[procnum]['score'] = grid.score(X_test,y_test)
    # return_dict[procnum]['grid']=grid



#將資料分成訓練組及測試組
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA

if __name__ == '__main__':
    # freeze_support()

    data[["Attribute8","Attribute10"]]=data[["Attribute8","Attribute10"]].applymap(convert_wind)

    data[["Attribute16","Attribute17"]]=data[["Attribute16","Attribute17"]].applymap(lambda x: 0 if x=="No" else 1)

    data=data.fillna(0)
    train_para={}

    X = data[["Attribute2","Attribute3","Attribute4","Attribute5","Attribute6","Attribute7","Attribute8","Attribute9","Attribute10","Attribute11","Attribute12","Attribute13","Attribute14","Attribute15","Attribute16"]]
    y = data['Attribute17']

    d_arr=[]

    pca = PCA(n_components=6)
    X_reduced = pca.fit_transform(X)

    manager = mp.Manager()
    
    i=0
    for C,gamma in product([1,10,100,1000],[1,0.1,0.01,0.0001]):

        p_list = []
        score_list=[]
        train_para[i]={}
        train_para[i]["accuracy"]=[]
        return_dict = manager.dict()
        
        train_para[i]["C"]=C
        train_para[i]["gamma"]=gamma
        for j in range(5):
            p=mp.Process(target=train_module,args=(X_reduced,y,C,gamma,return_dict,j))
            p_list.append(p)
            p.start()

        for k in range(5):
            p_list[k].join()    

        print(return_dict.values())
        
        print(pd.DataFrame(return_dict.values())['score'].mean())
        train_para[i]["accuracy"]=pd.DataFrame(return_dict.values())['score'].mean()
        train_para[i]['grid']=return_dict[0]["grid"]
        i+=1
        
    print(train_para)
    train_para=pd.DataFrame(train_para.values())
    print(train_para)


    #利用剛剛的最佳參考再重新預測測試組
    # grid_predictions = grid.predict(test_data[["Attribute2","Attribute3","Attribute4","Attribute5","Attribute6","Attribute7","Attribute8","Attribute9","Attribute10","Attribute11","Attribute12","Attribute13","Attribute14","Attribute15","Attribute16"]])


    test_data=pd.read_csv("test.csv")
    test_data[["Attribute8","Attribute10"]]=test_data[["Attribute8","Attribute10"]].applymap(convert_wind)

    test_data[["Attribute16"]]=test_data[["Attribute16"]].applymap(lambda x: 0 if x=="No" else 1)

    test_data=test_data.fillna(0)
    a=0
    ans=list(pd.read_csv("ex_submit.csv")["ans"])
    input=test_data[["Attribute2","Attribute3","Attribute4","Attribute5","Attribute6","Attribute7","Attribute8","Attribute9","Attribute10","Attribute11","Attribute12","Attribute13","Attribute14","Attribute15","Attribute16"]]
    
    pca = PCA(n_components=6)
    input_reduced = pca.fit_transform(input)
    for grid in train_para['grid']:
        print(grid.score(input_reduced,ans))
        # train_para['result'][a]=grid.score(input,ans)

    


    # from sklearn.metrics import classification_report, confusion_matrix
    # #評估新參考的預測結果好壞
    # print('confusion_matrix : ')
    # print(confusion_matrix(ans,grid_predictions))
    # print('classification_report : ')
    # print(classification_report(ans,grid_predictions))