from numpy import loadtxt
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import os

# load data
dataset = loadtxt(os.path.join('data', 'diabetes.csv'), delimiter=",", skiprows=1)

# split data into X and y
X = dataset[:, 0:8]
Y = dataset[:, 8]

# split data into train and test sets
seed = 7
test_size = 0.33
X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=test_size,
                                                    random_state=seed)

# fit model on training data
model = XGBClassifier()
model.fit(X_train, y_train)

print(model)

# make predictions for test data
predictions = model.predict(X_test)

# evaluate predictions
accuracy = accuracy_score(y_test, predictions)
print("Accuracy: %.2f%%" % (accuracy * 100.0))
