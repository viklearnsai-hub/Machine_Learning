import numpy
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

from sklearn.datasets import load_digits
from sklearn.metrics import accuracy_score

data = load_digits()

x = data.data
y = data.target

x_train,x_test,y_train,y_test = train_test_split(x,y,random_state=42,test_size=0.4)

params = {
    "C":[0.01,0.1,1,10,100],
    "solver":["liblinear","lbfgs","saga"],
    "penalty":["l2"]
}


model = LogisticRegression(multi_class="ovr",solver="lbfgs")

model.fit(x_train,y_train)

y_pred = model.predict(x_test)

acc = accuracy_score(y_test,y_pred)

print("Test",y_test[:20])
print("Pred",y_pred[:20])
print("Accuracy is ",acc)