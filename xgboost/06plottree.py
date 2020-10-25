# plot decision tree
from numpy import loadtxt
from xgboost import XGBClassifier
from xgboost import plot_tree
from matplotlib import pyplot

# load data
dataset = loadtxt('data/diabetes.csv', delimiter=",", skiprows=1)
# split data into X and y
X = dataset[:, 0:8]
y = dataset[:, 8]
# fit model on training data
model = XGBClassifier()
model.fit(X, y)
# plot single tree
# plot_tree(model)
# pyplot.show()
# plot_tree(model, num_trees=4)
# pyplot.show()
plot_tree(model, num_trees=0, rankdir='LR')
pyplot.show()
