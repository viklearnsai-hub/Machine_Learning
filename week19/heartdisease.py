import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
data = pd.read_csv("res\week19\heart.csv",index_col=0)


y= data["HeartDisease"]
x = data.drop("HeartDisease",axis=1)
x_encoded = pd.get_dummies(x)
x_train,x_test,y_train,y_test = train_test_split(x_encoded,y,test_size=0.3,random_state=42)

knns = [3,5,7,9]

for k in knns:
    eculadianModel = KNeighborsClassifier(n_neighbors=k)
    eculadianModel.fit(x_encoded,y)

    y_pred = eculadianModel.predict(x_test)
    print(f"Euclidean (k={k}): Accuracy = {accuracy_score(y_test, y_pred):.3f}")

    manhattanModel = KNeighborsClassifier(n_neighbors=k,metric="manhattan")
    manhattanModel.fit(x_encoded,y)

    y_pred = manhattanModel.predict(x_test)
    print(f"Manhattan (k={k}): Accuracy = {accuracy_score(y_test, y_pred):.3f}")
    print("-"*50)