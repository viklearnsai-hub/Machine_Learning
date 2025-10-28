"""
1: Kama
2: Rosa
3: Canadian
"""


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report


urlOfDataSet = "https://archive.ics.uci.edu/ml/machine-learning-databases/00236/seeds_dataset.txt"

columnNames = ["area", "perimeter", "compactness"," length_of_kernel", "width_of_kernel", "asymmetry_coefficient", "length_of_groove", "class"]

df = pd.read_csv(urlOfDataSet,names=columnNames,sep="\s+")

x = df.drop("class",axis=1)
y = df["class"]


x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.3,random_state=42)

params = {
    "C":[0.01,0.1,1,10,100],
    "solver":["liblinear","lbfgs","saga"],
    "penalty":["l2"]
}

grid = GridSearchCV(LogisticRegression(multi_class="ovr"),params,cv=5)
grid.fit(x_train,y_train)

y_pred = grid.predict(x_test)


accuracy = accuracy_score(y_test, y_pred)

print("ACCURACCY IS",accuracy)