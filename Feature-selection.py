import pandas as pd
import numpy as np
from io import StringIO
#from sklearn.preprocessing import Imputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.base import clone
from sklearn.metrics import accuracy_score
from itertools import combinations
import matplotlib.pyplot as plt

# for sklearn 0.18's alternative syntax
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.grid_search import train_test_split
else:
    from sklearn.model_selection import train_test_split

    #############################################################################
    print(50 * '=')
    print('Section: Partitioning a dataset in training and test sets')
    print(50 * '-')

    df_wine = pd.read_csv('https://archive.ics.uci.edu/'
                          'ml/machine-learning-databases/wine/wine.data',
                          header=None)

    df_wine.columns = ['Class label', 'Alcohol', 'Malic acid', 'Ash',
                       'Alcalinity of ash', 'Magnesium', 'Total phenols',
                       'Flavanoids', 'Nonflavanoid phenols', 'Proanthocyanins',
                       'Color intensity', 'Hue', 'OD280/OD315 of diluted wines',
                       'Proline']

    print('Class labels', np.unique(df_wine['Class label']))

    print('\nWine data excerpt:\n\n', df_wine.head())

    X, y = df_wine.iloc[:, 1:].values, df_wine.iloc[:, 0].values

    X_train, X_test, y_train, y_test = \
        train_test_split(X, y, test_size=0.3, random_state=0)



stdsc = StandardScaler()
X_train_std = stdsc.fit_transform(X_train)
X_test_std = stdsc.transform(X_test)


#############################################################################
print(50 * '=')
print('Section: Sequential feature selection algorithms')
print(50 * '-')


class SBS():
    def __init__(self, estimator, k_features, scoring=accuracy_score,
                 test_size=0.25, random_state=1):
        self.scoring = scoring
        self.estimator = clone(estimator)
        self.k_features = k_features
        self.test_size = test_size
        self.random_state = random_state

    def fit(self, X, y):

        X_train, X_test, y_train, y_test = \
            train_test_split(X, y, test_size=self.test_size,
                             random_state=self.random_state)

        dim = X_train.shape[1]
        print('dim', dim)
        self.indices_ = tuple(range(dim))
        print('indices', self.indices_)
        self.subsets_ = [self.indices_]
        print('subsets', self.subsets_)
        score = self._calc_score(X_train, y_train,
                                 X_test, y_test, self.indices_)
        self.scores_ = [score]

        while dim > self.k_features:
            scores = []
            subsets = []

#below is all the possible 12 digit combinations between 0 and 12 numbers...
# then 0 and 11 numbers, then 0 and 10 numbers and so on until k( which is 1 in this example)
#and it tests the score_accuracy upon each combination
            for p in combinations(self.indices_, r=dim - 1):
     #           print("P", p)
                score = self._calc_score(X_train, y_train,
                                         X_test, y_test, p)
#scores ში აგროვებს თითუელი კომბინაციის accuracy scores, მაგრად დაუკვირდი ეს სხვა scores არის, ქვემოთ არის კიდევ ერთი scores_ რომელიც საუკეთესოს accuracyს აგროვებს
                scores.append(score)
#subset აგროვებს კომბინაციებს
                subsets.append(p)
    #        print("scores", scores)
#აქ უნდა იყო ფრთხილად, best არის არა სვეტი არამედ ის კომბინაცია რომელშიც სვეტი N4 არ ფიგურირებს და შესაბამისად აძლევს ყველაზე დიდ prediction scores, შესაბამისად ეს იყო მე [8] კომბინაცია
            best = np.argmax(scores)
            print("scores", scores)
            print("best", best)
#indece ში იწერება ის კომბინაცია რომელსაც საუკეთესო prediction accuracy ჰქონდა, რაც გულისხმობს იმ 1 სვეტის ამოგდებას და ახალ ციკლში ის 1 სვეტი აღარ იფიგურირებს
            self.indices_ = subsets[best]
            self.subsets_.append(self.indices_)
  #          print("indices:", self.indices_, "subsets", self.subsets_)
            dim -= 1

#ეს scores_ ს უკვე აგროვებს საუკეთესოს და გამოაქვს plot ზე
            self.scores_.append(scores[best])
            print("scores", self.scores_)
        self.k_score_ = self.scores_[-1]
        print("self.k_score", self.k_score_)

        return self

    def transform(self, X):
        return X[:, self.indices_]

    def _calc_score(self, X_train, y_train, X_test, y_test, indices):
        self.estimator.fit(X_train[:, indices], y_train)
        y_pred = self.estimator.predict(X_test[:, indices])
        score = self.scoring(y_test, y_pred)
        #below is the call to function accuracy_score as defined in init
   #     print('scoring', score)
        return score


knn = KNeighborsClassifier(n_neighbors=2)
# A Python program to print all combinations
# with an element-to-itself combination is
# also included
from itertools import combinations_with_replacement

# Get all combinations of [1, 2, 3] and length 2
comb = combinations_with_replacement([1, 2, 3], 2)

# Print the obtained combinations
for i in list(comb):
    print(i)
# selecting features
sbs = SBS(knn, k_features=1)
sbs.fit(X_train_std, y_train)

# plotting performance of feature subsets
k_feat = [len(k) for k in sbs.subsets_]
print("k_feat", k_feat, "sbs.scores", sbs.scores_)

plt.plot(k_feat, sbs.scores_, marker='o')
plt.ylim([0.7, 1.1])
plt.ylabel('Accuracy')
plt.xlabel('Number of features')
plt.grid()
# plt.tight_layout()
# plt.savefig('./sbs.png', dpi=300)
plt.show()


k5 = list(sbs.subsets_[8])
print('Selected top 5 features:\n', df_wine.columns[1:][k5])

knn.fit(X_train_std, y_train)
print('\nPerformance using all features:\n')
print('Training accuracy:', knn.score(X_train_std, y_train))
print('Test accuracy:', knn.score(X_test_std, y_test))

knn.fit(X_train_std[:, k5], y_train)
print('\nPerformance using the top 5 features:\n')
print('Training accuracy:', knn.score(X_train_std[:, k5], y_train))
print('Test accuracy:', knn.score(X_test_std[:, k5], y_test))