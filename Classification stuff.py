a
# coding: utf-8

# In[1]:

from sklearn import datasets
#import imp_fil_mat
import scipy.io as sio
#sio.matlab

import numpy as np


import scipy.io as sio
sio.matlab





# In[2]:

featmat = sio.loadmat('NUMPYTESTFEATURES.mat')
#ww = sio.whosmat('NUMPYTESTFEATURES.mat')
#print ww

xtrain = featmat['xtrain']
xtest = featmat['xtest']
ytrain = featmat['ytrain']
ytest = featmat['ytest']


# In[3]:

ytrain1 = np.ravel(ytrain)
ytest1 = np.ravel(ytest)


# In[4]:

#from sklearn import svm
from sklearn.svm import SVC

clf = SVC()
clf.fit(xtrain,ytrain1)
dec = clf.score(xtest,ytest1)
#results = clf.score(xtest,ytest)
print dec


# In[ ]:

#logistic regression? as alternative to lasso? sklearns feature selection modules say that lasso is for regression, and 
# logistic regression and linear svc are for classification'
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.feature_selection import SelectFromModel as sfm


# In[ ]:


lr1 = logreg()
lr1.fit(xtrain,ytrain1)

print lr1.fit(xtrain,ytrain1)
print (lr1.score(xtest,ytest1))
print lr1.coef_
model = sfm(lr1,prefit=True)
xnew = model.transform(xtrain)
print xtrain.shape
print xnew.shape
#lr2 = logreg()
#lr2.fit(xnew,ytrain1)
print xtrain
print xnew


# In[ ]:


lr2 = logreg(penalty = "l1")
lr2.fit(xtrain,ytrain1)

print (lr2.score(xtest,ytest1))
#print lr1.coef_
model2 = sfm(lr2,prefit=True)
xnew1 = model2.transform(xtrain)
# maybe instead write xnew1 = sfm(lr2,prefit=True).transform(xtrain)
# #lr2 = logreg()
# #lr2.fit(xnew,ytrain1)
# print xtrain
# print xnew
print xtrain.shape
print xnew1.shape
# took forever, but at least if we run it once, we will know what features we should use?


# In[ ]:

# feature selection usually done as pre-processing, before actual learning, use sklearn.pipeline.Pipeline:
clf = Pipeline([('feature_selection',sfm(LinearSVC(penalty = "l1")),'classification',RandomForestClassifier())])
clf.fit(xtrain,ytrain1)


# In[ ]:

# lr3 = logreg(C = .5)
# lr3.fit(xtrain,ytrain1)

# print (lr3.score(xtest,ytest1))
# #print lr1.coef_
# model3 = sfm(lr3,prefit=True)
# xnew3 = model3.transform(xtrain)
# print xtrain.shape
# print xnew3.shape



# In[ ]:

lr4 = logreg(C = .5, penalty = "l1")
lr4.fit(xtrain,ytrain1)

print (lr4.score(xtest,ytest1))
#print lr1.coef_
model4 = sfm(lr4,prefit=True)
xnew4 = model4.transform(xtrain)
print xtrain.shape
print xnew4.shape


# In[5]:

# implementing with kernels
svc2 = SVC(kernel = 'rbf')
svc2.fit(xtrain,ytrain1)
print (svc2.score(xtest,ytest1))
# took a wile, got the same as the plain svc


# In[ ]:




# In[6]:

# svc3 = SVC(kernel = 'poly')
# svc3.fit(xtrain,ytrain1)
# print (svc3.score(xtest,ytest1))
# # took longer than the rbf kernel, so i just stopped it


# In[7]:

# #NuSVC - similar to SVC, but uses a parameter to control the number of support vectors
# from sklearn.svm import NuSVC
# Nsvc = NuSVC()
# Nsvc.fit(xtrain,ytrain1)
# print(Nsvc.score(xtest,ytest1))
# # also taking a long time to run, probably not worth it?


# In[8]:

#from sklearn.svm import LinearSVC
# lin_clf = LinearSVC()
# lin_clf.fit(xtrain,ytrain1)
# dec1 = lin_clf.score(xtest,ytest1)
#print dec1
# this took a ridiculous amount of time, and only got 49%


# In[9]:

from sklearn import linear_model
i = .1
lassoclf = linear_model.Lasso(alpha = i)
#lassoclf.fit(xtrain,ytrain)
#lassoclf.coef_
#dec2 = lassoclf.score(xtest,ytest)
#print dec2


# In[ ]:

lassoclf.fit(xtrain,ytrain)


# In[ ]:

lassoclf.coef_


# In[ ]:

dec2 = lassoclf.score(xtest,ytest)
dec2


# In[ ]:

from sklearn import linear_model
for i in [0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]:
    lassoclf = linear_model.Lasso(alpha = i)
    lassoclf.fit(xtrain,ytrain1)
    lassoclf.coef_
    dec2 = lassoclf.score(xtest,ytest1)
    print dec2


# In[ ]:

get_ipython().magic(u'whos')


# In[ ]:




# In[10]:

from sklearn import neighbors
#from sklearn.neighbors import NearestNeighbors
knn1 = neighbors.KNeighborsClassifier() # this classifies based on the #k nearest neighbors, where k is definted by the user.
knn1.fit(xtrain,ytrain1)
print(knn1.score(xtest,ytest1))
# k =5 is the default
# the algorithm parameter is usable, as is the weight function


# In[11]:

knn2 = neighbors.KNeighborsClassifier(n_neighbors = 10)
knn2.fit(xtrain,ytrain1)
print(knn2.score(xtest,ytest1))


# In[12]:

knn3 = neighbors.KNeighborsClassifier(n_neighbors = 3)
knn3.fit(xtrain,ytrain1)
print(knn3.score(xtest,ytest1))


# In[13]:

# can also change the algorithm parameter, defaulted to auto, which automaticall selects the appropriate algorithm based on the values pass to fit
knn4 = neighbors.KNeighborsClassifier(algorithm = 'ball_tree')
knn4.fit(xtrain,ytrain1)
print(knn4.score(xtest,ytest1))


# In[14]:

knn5 = neighbors.KNeighborsClassifier(algorithm = 'kd_tree')
knn5.fit(xtrain,ytrain1)
print(knn5.score(xtest,ytest1))


# In[15]:

# DO NOT RUN THIS, BROKE COMPUTER...
# knn6 = neighbors.KNeighborsClassifier(algorithm = 'brute')
# knn6.fit(xtrain,ytrain1)
# print(knn6.score(xtest,ytest1))
# can also change leaf_size (Default 30),metrix,p (power parameter), and a few others


# In[16]:

from sklearn.neighbors import RadiusNeighborsClassifier
# This also didn't work, so don't use
# rnc1 = RadiusNeighborsClassifier()
# #default is r = 1.0
# rnc1.fit(xtrain,ytrain1)
# print (rnc1.score(xtest,ytest1))


# In[ ]:

get_ipython().magic(u'whos')


# In[17]:

# Nearest centroid
from sklearn.neighbors.nearest_centroid import NearestCentroid
ncc1 = NearestCentroid()
ncc1.fit(xtrain,ytrain1)
print (ncc1.score(xtest,ytest1))


# In[18]:

# Nearest shrunken Centroid
for shrinkage in [None,0.05,0.1,0.2,0.3,0.4,0.5]:
    ncc2 = NearestCentroid(shrink_threshold = shrinkage)
    ncc2.fit(xtrain,ytrain1)
    print(ncc2.score(xtest,ytest1))


# In[19]:

# linear discriminant analysis - classifier with linear decision boundary - 
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as linda
lda = linda()
lda.fit(xtrain,ytrain1)
print(lda.score(xtest,ytest1))


# In[20]:

# Quadradic discriminant analysis - classifier with quadratic decision boundary - 
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as quadda
qda = quadda()
qda.fit(xtrain,ytrain1)
print(qda.score(xtest,ytest1))

# might want to try normalizing stuff, or trying to fix rank deficiencies?


# In[ ]:

from numpy.linalg import matrix_rank
matrix_rank(xtrain)
xtrain.shape


# In[ ]:




# In[21]:

### TREESSSSS
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier as DTC
tree1 = DTC()
print tree1
tree1.fit(xtrain,ytrain1)
print tree1.fit(xtrain,ytrain1)
print tree1.score(xtest,ytest1)


# In[22]:

from sklearn.tree import ExtraTreeClassifier as ETC
tree2 = ETC()
print tree2
tree2.fit(xtrain,ytrain1)
print tree2.fit(xtrain,ytrain1)
print tree2.score(xtest,ytest1)


# In[23]:

from sklearn.ensemble import BaggingClassifier
bagging1 = BaggingClassifier(ETC())
bagging1.fit(xtrain,ytrain1)
print bagging1.score(xtest,ytest1)


# In[24]:

from sklearn.ensemble import BaggingClassifier
bagging2 = BaggingClassifier(ETC(),bootstrap=False,bootstrap_features=False)
bagging2.fit(xtrain,ytrain1)
print bagging2.score(xtest,ytest1)


# In[25]:

from sklearn.ensemble import RandomForestClassifier as RFC
tree3 = RFC()
tree3.fit(xtrain,ytrain1)
print tree3.score(xtest,ytest1)


# In[26]:

from sklearn.ensemble import BaggingClassifier
bagging3 = BaggingClassifier(RFC(),bootstrap=False,bootstrap_features=False)
bagging3.fit(xtrain,ytrain1)
print bagging3.score(xtest,ytest1)


# In[27]:

from sklearn.ensemble import BaggingClassifier
bagging4 = BaggingClassifier(SVC(),bootstrap=False,bootstrap_features=False)
bagging4.fit(xtrain,ytrain1)
print bagging4.score(xtest,ytest1)


# In[28]:

from sklearn.ensemble import RandomForestClassifier as RFC
tree4 = RFC(bootstrap=False)
tree4.fit(xtrain,ytrain1)
print tree4.score(xtest,ytest1)


# In[29]:

from sklearn.ensemble import AdaBoostClassifier as ABC
tree5 = ABC()
tree5.fit(xtrain,ytrain1)
print(tree5.score(xtest,ytest1))


# In[30]:

from sklearn.ensemble import GradientBoostingClassifier as GBC
tree6 = GBC()
tree6.fit(xtrain,ytrain1)
print(tree6.score(xtest,ytest1))
# look at n_estimators and change that along with changing warmstart to be true


# In[31]:

# votingClassifiers combine completely different machine learning classifiers and use a majority vote
clff1 = SVC()
clff2 = RFC(bootstrap=False)
clff3 = ETC()
clff4 = neighbors.KNeighborsClassifier()
clff5 = quadda()
from sklearn.ensemble import VotingClassifier
from sklearn import cross_validation
eclf = VotingClassifier(estimators = [('svc',clff1),('rfc',clff2),('etc',clff3),('knn',clff4),('qda',clff5)])
eclf = eclf.fit(xtrain,ytrain1)
print(eclf.score(xtest,ytest1))
# for claf, label in zip([clff1,clff2,clff3,clff4,clff5,eclf],['SVC','RFC','ETC','KNN','QDA','Ensemble']):
#     cla
#     scores = crossvalidation.cross_val_score(claf,xtrain,ytrain1,scoring='accuracy')
#     print ()
    


# In[ ]:



