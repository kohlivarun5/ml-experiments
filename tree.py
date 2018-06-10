#!/usr/local/bin/python3
from sklearn import tree

data = [
    ["USD",3,"M","Quarterly","US0003M"],
    ["USD",6,"M","Semi-Annual","US0006M"],
    ["USD",12,"M","Annual","US0012M"],
    ["MXN",28,"D","28Days","MXNI"],
    ["USD",6,"M","Semi-Annual","US0003M"],
]

labels = [
    ["USD","EUR","MXN"],
    [3,6,12,28],
    ["M","D"],
    ["Quarterly","Semi-Annual","Annual","28Days"],
    ["US0003M","US0006M","US0012M","MXNI"]
]

def toIndexed(data,labels,indexsOfTarget,featureCols=None):
    X=[]
    Y = []
    for points in data:
        d=[]
        ys=""
        for i,p in enumerate(points):
            if i not in indexsOfTarget and (featureCols is None or i in featureCols):
                d.append(labels[i].index(p))
            elif i in indexsOfTarget :
                ys+=str(p)
        X.append(d)
        Y.append(ys)
    return X,Y

X,Y=toIndexed(data,labels,[4])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
print(clf.predict([X[0],X[3],X[4]]))

X,Y=toIndexed(data,labels,[3],[4])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
print(clf.predict([X[0],X[3],X[4]]))

X,Y=toIndexed(data,labels,[1,2],[3])
clf = tree.DecisionTreeClassifier()
clf = clf.fit(X, Y)
print(clf.predict([X[0],X[3],X[4]]))
