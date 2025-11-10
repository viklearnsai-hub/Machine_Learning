from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
X = [[1,10],
     [1,15],
     [2,10],
     [2,20],
     [3,30],
     [4,40]
    ]


"""
C - 1
B - 2
J - 3
"""
Y = ["chips","chips","biscuit","biscuit","juice","juice"]

k = 3


x_train,x_test,y_train,y_test = train_test_split(X,Y,test_size=0.3,random_state=18)

eculadianModel = KNeighborsClassifier(n_neighbors=k)
eculadianModel.fit(X,Y)

y_pred = eculadianModel.predict(x_test)
print(f"Model accuracy for Eculadian {accuracy_score(y_test,y_pred)}")



manhattanModel = KNeighborsClassifier(n_neighbors=k,metric="manhattan")
manhattanModel.fit(X,Y)

y_pred_m = eculadianModel.predict(x_test)
print(f"Model accuracy for Manhattan {accuracy_score(y_test,y_pred_m)}")


newStudent = [[2,18]]
pred = eculadianModel.predict(newStudent)

print(f"Student will purchase {pred[0]}")