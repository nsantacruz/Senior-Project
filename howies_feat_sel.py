import scipy.io as sio
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import SelectFromModel
import FinalClassifier
from sklearn.svm import SVC
from sklearn.feature_selection import RFE

xtrain_pow = sio.loadmat('xtrain_all_pow.mat')
xtrain_pow = xtrain_pow['xtrain']
ytrain_pow = sio.loadmat('ytrain_all_pow.mat')
ytrain_pow = ytrain_pow['ytrain']
xtesting = sio.loadmat('xtesting.mat')
xtesting = xtesting['xtesting']

print "Before feature selection..."
print xtrain_pow.shape
print ytrain_pow.shape
results_before = FinalClassifier.myclassify_practice_set(3, xtrain_pow, ytrain_pow, xtesting)
print results_before

clf = ExtraTreesClassifier()
ytrain_pow = np.ravel(ytrain_pow)
clf = clf.fit(xtrain_pow, ytrain_pow)
model = SelectFromModel(clf, prefit=True)
support = model.get_support()
xtesting_fsel_tree = xtesting[:,support]
xtrain_fsel_tree = model.transform(xtrain_pow)
print "After decision tree feature selection..."
print xtrain_fsel_tree.shape
results_after = FinalClassifier.myclassify_practice_set(3, xtrain_fsel_tree, ytrain_pow, xtesting_fsel_tree)
print results_after



