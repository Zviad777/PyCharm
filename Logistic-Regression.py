import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
import os
import numpy as np
from sklearn import datasets
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
# from sklearn.tree import export_graphviz
from matplotlib.colors import ListedColormap
import matplotlib.pyplot as plt
import warnings
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.grid_search import train_test_split
else:
    from sklearn.model_selection import train_test_split

print(50 * '=')
print('Section: First steps with scikit-learn')
print(50 * '-')

iris = datasets.load_iris()
X = iris.data[:, [2, 3]]
y = iris.target
print('Class labels:', np.unique(y))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=1, stratify=y)

class LogisticRegressionGD(object):

 def __init__(self, eta=0.05, n_iter=100, random_state=1):
  self.eta = eta
  self.n_iter = n_iter
  self.random_state = random_state

 def fit(self, X, y):
  rgen = np.random.RandomState(self.random_state)
  self.w_ = rgen.normal(loc=0.0, scale=0.01, size=1 + X.shape[1])
  print('self.w', self.w_)
  self.cost_ = []
  for i in range(self.n_iter):
   net_input = self.net_input(X)
   output = self.activation(net_input)
   errors = (y - output)
   self.w_[1:] += self.eta * X.T.dot(errors)
   self.w_[0] += self.eta * errors.sum()
   # note that we compute the logistic 'cost' now
   # instead of the sum of squared errors cost
   cost = (-y.dot(np.log(output)) - ((1 - y).dot(np.log(1 - output))))
   self.cost_.append(cost)
   print('Cost', self.cost_)
  return self

 def net_input(self, X):
  return np.dot(X, self.w_[1:]) + self.w_[0]

 def activation(self, z):
  return 1. / (1. + np.exp(-np.clip(z, -250, 250)))

 def predict(self, X):
  return np.where(self.net_input(X) >= 0.0, 1, 0)

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):

    # setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])

    # plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                           np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    #plt.contourf ხატავს მთელს სიბრტყეზე Z Prediction ის მიხედვით რომელი წერტილი რომელ კლასს მიეკუთვნება
    plt.contourf(xx1, xx2, Z, alpha=0.4, cmap=cmap)
    plt.xlim(xx1.min(), xx1.max())
    plt.ylim(xx2.min(), xx2.max())

    for idx, cl in enumerate(np.unique(y)):
        print ('idx', idx, 'cl', cl)
        # კლასიფიკაციას ახდენს  სადაც y უდრის 0ს, 1ს, და 2ს, 50-50-50 და სვამს შესაბამის წერტილებს სიბრტყეზე
        print('X', X[y == cl, 0], 'Y', X[y == cl, 1])
        plt.scatter(x=X[y == cl, 0], y=X[y == cl, 1],
                    alpha=0.8, c=cmap(idx),
                    marker=markers[idx], label=cl)

    # highlight test samples
    if test_idx:
        # plot all samples
        if not versiontuple(np.__version__) >= versiontuple('1.9.0'):
            X_test, y_test = X[list(test_idx), :], y[list(test_idx)]
            warnings.warn('Please update to NumPy 1.9.0 or newer')
        else:
            X_test, y_test = X[test_idx, :], y[test_idx]

        plt.scatter(X_test[:, 0],
                    X_test[:, 1],
                    c='',
                    edgecolor='black',
                    alpha=1.0,
                    linewidths=1,
                    marker='o',
                    s=100, label='test set')


# plt.tight_layout()
# plt.savefig('./figures/iris_perceptron_scikit.png', dpi=300)
plt.show()

X_train_01_subset = X_train[(y_train == 0) | (y_train == 1)]
y_train_01_subset = y_train[(y_train == 0) | (y_train == 1)]
lrgd = LogisticRegressionGD(eta=0.05, n_iter=1000, random_state=1)
lrgd.fit(X_train_01_subset, y_train_01_subset)
plot_decision_regions(X=X_train_01_subset, y=y_train_01_subset, classifier=lrgd)
plt.xlabel('petal length [standardized]')
plt.ylabel('petal width [standardized]')
plt.legend(loc='upper left')
plt.tight_layout()
plt.show()