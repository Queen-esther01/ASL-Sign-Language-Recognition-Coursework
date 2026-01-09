import seaborn as sns; import numpy as np; import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import os, ast; from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import matplotlib.pyplot as plt
from collections import Counter
from sklearn.svm import SVC


class KNN_Manual:
        def __init__(self,k=5): self.k=k
        def fit(self,X,y):
                self.X_train=X; self.y_train=y
        def predict(self,X_test):
                return np.array([ self._predict(x) for x in X_test ])
        def _predict(self,x):
                d=[np.sqrt(np.sum((x-x_t)**2)) for x_t in self.X_train]
                idx=np.argsort(d)[:self.k]
                labs=[ self.y_train[i] for i in idx ]
                return Counter(labs).most_common(1)[0][0]



csv_path=os.path.join("data","clean_dataset","data.csv")
if not os.path.exists(csv_path):
        print("Cannot find path",csv_path)
else:

        df=pd.read_csv(csv_path); df["landmark"]=df["landmark"].apply(ast.literal_eval)

        X_list=[]; y_list=[]
        for i,row in df.iterrows():
                f=np.array(row["landmark"]).flatten()
                if len(f)==63:
                        X_list.append(f); y_list.append(row["label"])

        X=np.array(X_list)
        enc=LabelEncoder()
        y=enc.fit_transform(y_list)
        class_names=enc.classes_


        # preprocessing
        X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)
        sc=StandardScaler()
        X_train_s=sc.fit_transform(X_train)
        X_test_s=sc.transform(X_test)

        print("Training models...")

        # KNN
        knn=KNN_Manual(k=5); knn.fit(X_train_s,y_train)
        knn_preds=knn.predict(X_test_s)

        # decision tree
        dt=DecisionTreeClassifier(max_depth=10,random_state=42)
        dt.fit(X_train,y_train); dt_preds=dt.predict(X_test)

        # svm
        svm=SVC(kernel="rbf",C=1.0)
        svm.fit(X_train_s,y_train); svm_preds=svm.predict(X_test_s)


        if not os.path.exists("reports"):
                os.makedirs("reports")

        # generation of confusion matrices
        for n,p in [("KNN_Manual",knn_preds),("Decision_Tree",dt_preds),("SVM",svm_preds)]:
                plt.figure(figsize=(8,6))
                sns.heatmap(confusion_matrix(y_test,p),annot=True,fmt="d",
                        xticklabels=class_names,yticklabels=class_names,cmap="Blues")
                plt.title("Confusion Matrix: "+n)
                plt.savefig("reports/"+n+"_cm.png")
                plt.close()
                print(n,"Accuracy:",accuracy_score(y_test,p))

        print("All complete. Files saved in reports folder.")
