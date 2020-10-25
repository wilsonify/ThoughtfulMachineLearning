# plot feature importance manually
from matplotlib import pyplot
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_importance

# load data
dataset = loadtxt('data/diabetes.csv', delimiter=",", skiprows=1)
# split data into X and y
X = dataset[:, 0:8]
y = dataset[:, 8]
# fit model on training data
model = XGBClassifier()
model.fit(X, y)
# feature importance
print(model.feature_importances_)
# plot
pyplot.bar(range(len(model.feature_importances_)), model.feature_importances_)
pyplot.show()

# plot feature importance
plot_importance(model)
pyplot.show()
