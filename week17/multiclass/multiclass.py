import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
# Load the Iris dataset
data = load_iris()
X = data.data
y = data.target
feature_names = data.feature_names



x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Create and train the Logistic Regression model
model = LogisticRegression()

model.fit(x_train, y_train)
# Make predictions on the test set

y_pred = model.predict(x_test)
# Evaluate the model's accuracy
accuracy = accuracy_score(y_test, y_pred)

print(classification_report(y_test, y_pred, target_names=data.target_names))