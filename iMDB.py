import pyprind
import pandas as pd
import os
import numpy as np
import re
import nltk
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.linear_model import SGDClassifier
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
import os
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RANSACRegressor
#from sklearn.cross_validation import train_test_split
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import Lasso
from sklearn.preprocessing import PolynomialFeatures
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation


# Added version check for recent scikit-learn 0.18 checks
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import GridSearchCV
else:
    from sklearn.model_selection import GridSearchCV

#############################################################################
""""
print(50 * '=')
print('Section: Obtaining the IMDb movie review dataset')
print(50 * '-')

print('!! This script assumes that the movie dataset is located in the'
      ' current directory under ./aclImdb')

_ = input('Please hit enter to continue.')

basepath = 'C:/Users/zjeiranashviliadm/PycharmProjects/untitled/venv/Scripts/aclImdb_v1'


labels = {'pos': 1, 'neg': 0}
pbar = pyprind.ProgBar(50000)
df = pd.DataFrame()
for s in ('test', 'train'):
    for l in ('pos', 'neg'):
        path = os.path.join(basepath, s, l)
        print("path", path)
        for file in sorted(os.listdir(path)):
            with open(os.path.join(path, file), 'r',
                      encoding='utf-8') as infile:
                txt = infile.read()
            df = df.append([[txt, labels[l]]], ignore_index=True)
            pbar.update()
df.columns = ['review', 'sentiment']
np.random.seed(0)
df = df.reindex(np.random.permutation(df.index))
df.to_csv('./movie_data.csv', index=False)

"""
df = pd.read_csv('movie_data.csv', encoding='utf-8')
print('Excerpt of the movie dataset', df.head(3))


#############################################################################
print(50 * '=')
print('Section: Transforming documents into feature vectors')
print(50 * '-')

count = CountVectorizer()
docs = np.array(['The sun is shining',
                 'The weather is sweet',
                 'The sun is shining and the weather is sweet'])
bag = count.fit_transform(docs)

print('Vocabulary', count.vocabulary_)
#პრინციპიმდგომარეობს იმაში რომ, რაც ნაკლებ დოკუმენტში გვხვდება სიტყვა მით მაღალ კოეფიციენტს ანიჭებს,
# რადგან მეტი შანსია უნიფიცირებული Label ქონდეს და უნიკალური იყოს, იგივე პონია როგორც 127 set of stop-words,
# ასეთ კატეგორიაში ხვდებიან ძირითადად სიტყვები როგორებიცაა: is and has like, და ა.შ
print('bag.toarray()', bag.toarray())

#############################################################################

print(50 * '=')
print('Section: Assessing word relevancy via term frequency-inverse'
      ' document frequency')
print(50 * '-')

np.set_printoptions(precision=2)
tfidf = TfidfTransformer(use_idf=True, norm='l2', smooth_idf=True)
print(tfidf.fit_transform(count.fit_transform(docs)).toarray())

tf_is = 2
n_docs = 3
idf_is = np.log((n_docs + 1) / (3 + 1))
tfidf_is = tf_is * (idf_is + 1)
print('tf-idf of term "is" = %.2f' % tfidf_is)


tfidf = TfidfTransformer(use_idf=True, norm=None, smooth_idf=True)
raw_tfidf = tfidf.fit_transform(count.fit_transform(docs)).toarray()[-1]
print('raw tf-idf', raw_tfidf)

l2_tfidf = raw_tfidf / np.sqrt(np.sum(raw_tfidf**2))
l2_tfidf
print('l2 tf-idf', l2_tfidf)

print("df.size", df.shape)

############################################################################
print(50 * '=')
print('Section: Cleaning text data')
print(50 * '-')

print('Excerpt:\n\n', df.loc[0, 'review'][-50:])


def preprocessor(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text)
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    return text


print('Preprocessor on Excerpt:\n\n', preprocessor(df.loc[0, 'review'][-50:]))

res = preprocessor("</a>This :) is :( a test :-)!")
print('Preprocessor on "</a>This :) is :( a test :-)!":\n\n', res)

df['review'] = df['review'].apply(preprocessor)


#############################################################################
print(50 * '=')
print('Section: Processing documents into tokens')
print(50 * '-')

porter = PorterStemmer()


def tokenizer(text):
    return text.split()


def tokenizer_porter(text):
    return [porter.stem(word) for word in text.split()]


t1 = tokenizer('runners like running and thus they run')
print("Tokenize: 'runners like running and thus they run'")
print(t1)

t2 = tokenizer_porter('runners like running and thus they run')
print("\nPorter-Tokenize: 'runners like running and thus they run'")
print(t2)

nltk.download('stopwords')


print('remove stop words')
stop = stopwords.words('english')

r = [w for w in tokenizer_porter('a runner likes running and runs a lot')[-10:]
     if w not in stop]

print(r)

""""
#############################################################################
print(50 * '=')
print('Section: Training a logistic regression model'
      ' for document classification')
print(50 * '-')


X_train = df.loc[:25000, 'review'].values
y_train = df.loc[:25000, 'sentiment'].values
X_test = df.loc[25000:, 'review'].values
y_test = df.loc[25000:, 'sentiment'].values


tfidf = TfidfVectorizer(strip_accents=None,
                        lowercase=False,
                        preprocessor=None)

param_grid = [{'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              {'vect__ngram_range': [(1, 1)],
               'vect__stop_words': [stop, None],
               'vect__tokenizer': [tokenizer, tokenizer_porter],
               'vect__use_idf':[False],
               'vect__norm':[None],
               'clf__penalty': ['l1', 'l2'],
               'clf__C': [1.0, 10.0, 100.0]},
              ]

lr_tfidf = Pipeline([('vect', tfidf),
                     ('clf', LogisticRegression(random_state=0))])

gs_lr_tfidf = GridSearchCV(lr_tfidf, param_grid,
                           scoring='accuracy',
                           cv=5,
                           verbose=1,
                           n_jobs=-1)

gs_lr_tfidf.fit(X_train, y_train)

print('Best parameter set: %s ' % gs_lr_tfidf.best_params_)
print('CV Accuracy: %.3f' % gs_lr_tfidf.best_score_)


clf = gs_lr_tfidf.best_estimator_
print('Test Accuracy: %.3f' % clf.score(X_test, y_test))
"""

#############################################################################
print(50 * '=')
print('Section: Working with bigger data - online'
      ' algorithms and out-of-core learning')
print(50 * '-')

stop = stopwords.words('english')


def tokenizer(text):
    text = re.sub('<[^>]*>', '', text)
    emoticons = re.findall('(?::|;|=)(?:-)?(?:\)|\(|D|P)', text.lower())
    text = re.sub('[\W]+', ' ', text.lower()) +\
        ' '.join(emoticons).replace('-', '')
    tokenized = [w for w in text.split() if w not in stop]
    return tokenized


def stream_docs(path):
    with open(path, 'r', encoding='utf-8') as csv:
        next(csv)  # skip header
        for line in csv:
#ეს ქვემოთ -3 და -2 სკარეე ვსევო cvs ში character ების წაკითხვის პარამეტრებია რა, მძიმის ამბავში, -1 ზე იმენა არაფერი არ არის, errors იძლევა
#example word[:2]   # character from the beginning to position 2 (excluded)
            text, label = line[:-3], int(line[-2])
            yield text, label

next(stream_docs(path='./movie_data.csv'))

def get_minibatch(doc_stream, size):
    docs, y = [], []
    try:
        for _ in range(size):
            text, label = next(doc_stream)
            docs.append(text)
            y.append(label)
    except StopIteration:
        return None, None
    return docs, y


vect = HashingVectorizer(decode_error='ignore',
                         n_features=2**21,
                         preprocessor=None,
                         tokenizer=tokenizer)

clf = SGDClassifier(loss='log', random_state=1)
#როგორც ჩანს ეს რადგან გენერატორზე მიმთითებელია იმახსოვრებს რეკურსიას სად გაჩერდა და ტესტი იქიდან აგრძელებს ბოლო 5000ზე ქვემოთ
doc_stream = stream_docs(path='./movie_data.csv')

pbar = pyprind.ProgBar(45)

classes = np.array([0, 1])
for _ in range(45):
    X_train, y_train = get_minibatch(doc_stream, size=1000)
    if not X_train:
        break
    X_train = vect.transform(X_train)
    clf.partial_fit(X_train, y_train, classes=classes)
    pbar.update()

X_test, y_test = get_minibatch(doc_stream, size=5000)
X_test = vect.transform(X_test)
print('Accuracy: %.3f' % clf.score(X_test, y_test))

clf = clf.partial_fit(X_test, y_test)



# Added version check for recent scikit-learn 0.18 checks
from distutils.version import LooseVersion as Version
from sklearn import __version__ as sklearn_version
if Version(sklearn_version) < '0.18':
    from sklearn.cross_validation import train_test_split
else:
    from sklearn.model_selection import train_test_split

df = pd.read_csv('movie_data.csv', encoding='utf-8')

count = CountVectorizer(stop_words='english', max_df=.1, max_features=5000)
X = count.fit_transform(df['review'].values)

# ეს სავარაუდოდ ეძებს 2 სიტყვას რომელიც ყველაზე ხშირად გვხვდება ერთად, შემდეგ მესამეს რომელიც ამათთან ერთად გვხვდება
# ყველაზე ხშირად შემდეგ მეოთხეს და ასე შემდეგ, მეერე ამას ანიჭებს 1 ტოპიკს და გადადის with replacement მეთოდით სხვა topic ზე

lda = LatentDirichletAllocation(n_components=10, random_state=123, learning_method='batch')

"""
# აქ პროსტა ვტესტავდი ქვემოთ რას აკეთებდა სკრიპტი
n_top_words = 5
array = [3,1,4,5,66,1,2,-3]
new = np.argsort(array)
print("new", new)

new = np.argsort(array)[:-n_top_words - 1:-1]
print("new", new)
"""


X_topics = lda.fit_transform(X)

n_top_words = 5
feature_names = count.get_feature_names()


for topic_idx, topic in enumerate(lda.components_):
 print("Topic %d:" % (topic_idx + 1))
 #აქ არაფერი განსაკუთრებული არ არის სინამდვილეში, უბრალოდ რამდენიმე ოპერაციია გაერთიანებული, კერძოდ:
 #topic.argsort() ახარისხებს ზრდადობით და ინდექსებს ინახავს values მაგივრად, ხოლო -n_top_words - 1 ეს არის
 #-6 მდე (ანუ 5 მნიშვნელობა ბოლოდან) ხოლო :-1 უკვე რევერსულად ალაგებს, რადგან ბოლოში ზრდადი მნიშვნელობებია
 print(" ".join([feature_names[i] for i in topic.argsort()[:-n_top_words - 1:-1]]))

dest = os.path.join('movieclassifier', 'pkl_objects')
if not os.path.exists(dest):
    os.makedirs(dest)

pickle.dump(stop, open(os.path.join(dest, 'stopwords.pkl'), 'wb'), protocol=4)
pickle.dump(clf, open(os.path.join(dest, 'classifier.pkl'), 'wb'), protocol=4)
