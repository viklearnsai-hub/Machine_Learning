"""
studyhours|pass/fail
1 | 0 [Fail]
2 | 0 [Fail]
3 | 0 [Fail]
4 | 0 [Fail]
5 | 1 [Pass]
6 | 1 [Pass]
7 | 1 [Pass]
8 | 1 [Pass]
9 | 1 [Pass]
10 | 1 [Pass]
"""

from sklearn.naive_bayes import BernoulliNB
import numpy as np
X = np.array([[1],[2],[3],[4],[5],[6],[7],[8],[9],[10]])
Y = np.array([0,0,0,0,1,1,1,1,1,1])

model = BernoulliNB()
model.fit(X,Y)

testData = [5]
result = model.predict(testData)
print("Pass" if result[0] == 1 else "Fail")

"""TODO Fix this"""