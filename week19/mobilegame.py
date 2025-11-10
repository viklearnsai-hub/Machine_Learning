"""
A group of students in a college spend different amounts of time studying and playing each day. Based on these activities, they tend to choose different types of games to relax.
Your task is to build a K-Nearest Neighbors (KNN) classification model that predicts the type of game a student is likely to choose based on two features:
Study Hours per Day
Play Time per Day (in Hours)
The possible game choices are:
Mobile Game
Chess
Cricket
You are given the following dataset:

Study Hours	Play Time	    Game
1	        5	            Mobile Game
2	        3	            Mobile Game
3	        2	            Chess
4	        1	            Chess
5	        3	            Cricket
6	        4           	Cricket
3.5	        3               ?


Study Hours	Play Time	    Game
1	        5	            Mobile Game	6.25	4	10.25	3.201562119
2	        3	            Mobile Game	2.25	0	2.25	1.5
3	        2	            Chess	0.25	1	1.25	1.118033989
4	        1	            Chess	0.25	4	4.25	2.061552813
5	        3	            Cricket	2.25	0	2.25	1.5
6	        4	            Cricket	6.25	1	7.25	2.692582404
3.5	        3

k = 3
2.449489743	Chess

"""

from sklearn.neighbors import KNeighborsClassifier

X =[[1,5],
    [2,3],
    [3,2],
    [4,1],
    [5,3],
    [6,4]]

Y = ["Mobile","Mobile","Chess","Chess","Cricket","Cricket"]

k = 3
model = KNeighborsClassifier(n_neighbors=k)
model.fit(X,Y)

testData = [[3.5,3]]
preds = model.predict(testData)

print(f"Test child will play the game {preds[0]}")