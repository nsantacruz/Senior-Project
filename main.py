
import classtest
import FinalClassifier
import scipy.io as sio
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import ExtraTreeClassifier as ETC


xtrain=sio.loadmat('xtrainNorm.mat')
xtrain=xtrain['xtrain']

ytrain = sio.loadmat('ytrainNorm.mat')
ytrain = ytrain['ytrain']

xtest = sio.loadmat('xtestNorm.mat')
xtest = xtest['xtest']

ytest=sio.loadmat('ytestNorm.mat')
ytest=ytest['ytest']

ytrain1 = np.ravel(ytrain)
ytest1 = np.ravel(ytest)



xtrainhold=sio.loadmat('xtrain2holdout.mat')
xtrainhold=xtrainhold['xtrain']

ytrainhold = sio.loadmat('ytrain2holdout.mat')
ytrainhold = ytrainhold['ytrain']

xtesthold = sio.loadmat('xtest2holdout.mat')
xtesthold = xtesthold['xtest']

ytesthold=sio.loadmat('ytest2holdout.mat')
ytesthold=ytesthold['ytest']

ytrainhold1 = np.ravel(ytrainhold)
ytesthold1 = np.ravel(ytesthold)


xtrainall=sio.loadmat('xtrainAll.mat')
xtrainall=xtrainall['xtrain']

ytrainall = sio.loadmat('ytrainAll.mat')
ytrainall = ytrainall['ytrain']

xtestall = sio.loadmat('xtesting.mat')
xtestall = xtestall['xtesting']

xtestallNorm = sio.loadmat('xtestingNorm.mat')
xtestallNorm = xtestallNorm['xtestingNorm']

ytrainall1 = np.ravel(ytrainall)



xtrunclength = sio.loadmat('xtrunclength.mat')
xtrunclength = xtrunclength['xtrunclength']
# print xtrainall.shape
# print ytrainall.shape
# print ytrainall1.shape
# print xtestall.shape

xtestall = xtestall[~np.isnan(xtestall).any(axis=1),:]

ytestclasses,yteststring = FinalClassifier.myclassify_practice_set(numfiers=1,xtrain=xtrainall,ytrain=ytrainall1,xtest=xtestall)
#classtest.myclassify(numfiers=5,xtrain=xtrain,ytrain=ytrain1,xtest=xtest,ytest=ytest1)

print xtrunclength.shape


for i in xtrunclength:
    if i==xtrunclength[0]:
        





# bagging2 = BaggingClassifier(ETC(),bootstrap=False,bootstrap_features=False)
# bagging2.fit(xtrainhold,ytrainhold1)
# #print bagging2.score(xtest,ytest)
#
# print "\n for original holdouts \n" + "on training set score was" + str(bagging2.score(xtrainhold,ytrainhold1))
# print "on holdout set score was" + str(bagging2.score(xtesthold,ytesthold1))
#
#
#
#
# bagging2 = BaggingClassifier(ETC(),bootstrap=False,bootstrap_features=False)
# bagging2.fit(xtrain,ytrain1)
# #print bagging2.score(xtest,ytest)
#
# print "for normalized signal \n" "on training set score was" + str(bagging2.score(xtrain,ytrain1))
# print "on holdout set score was" + str(bagging2.score(xtest,ytest1))

# bagging2 = BaggingClassifier(ETC(),bootstrap=False,bootstrap_features=False)
# bagging2.fit(xtrain,ytrain1)
# #print bagging2.score(xtest,ytest)
# ytest = bagging2.predict(xtest)
# print ytest[1:1000]


#classtest.myclassify(numfiers=21,xtrain=xtrain,ytrain=ytrain1,xtest=xtest,ytest=ytest1)

# xtrainwo = sio.loadmat('xtrainwo.mat')
# # print xtrainwo
# xtrainwo = xtrainwo['xtrain']
# ytrainwo = sio.loadmat('ytrainwo.mat')
# ytrainwo = ytrainwo['ytrain']
# xtestwo = sio.loadmat('xtestwo.mat')
# xtestwo = xtestwo['xtest']
# ytestwo = sio.loadmat('ytestwo.mat')
# ytestwo = ytestwo['ytest']
#
# ytrain1new = np.ravel(ytrainwo)
# ytest1new = np.ravel(ytestwo)
#
# # classtest.myclassify(numfiers=6,xtrain=xtrainwo,ytrain=ytrain1new,xtest=xtestwo,ytest=ytest1new)
#
# print xtrainwo.shape
# print xtestwo.shape
#
#







# from sklearn import datasets
# import imp_fil_mat
# import scipy.io as sio
# #sio.matlab
# from sklearn import svm
# from sklearn.svm import LinearSVC
# from sklearn.svm import SVC
#
# import numpy as np
# from sklearn import linear_model
#
# #import test
# #test.myfunc("hello")
#
#
#
# #imp_fil_mat.train_test()
# #xtrain,xtest,ytrain,ytest = imp_fil_mat.train_test()
#
#
# # MAKE THIS A FUNCTION!! having trouble
# import scipy.io as sio
# sio.matlab
# featmat = sio.loadmat('NUMPYTESTFEATURES.mat')
# #ww = sio.whosmat('NUMPYTESTFEATURES.mat')
# #print ww
# #print featmat
# xtrain = featmat['xtrain']
# xtest = featmat['xtest']
# ytrain = featmat['ytrain']
# ytest = featmat['ytest']
#
# # print xtrain.shape
# # print ytrain.shape
# # print xtest.shape
# # print ytest.shape
#
# ytrain1 = np.ravel(ytrain)
# ytest1 = np.ravel(ytest)
#
# clf = SVC()
# clf.fit(xtrain,ytrain1)
# dec = clf.score(xtest,ytest1)
# #results = clf.score(xtest,ytest)
# print dec
#
# # lin_clf = LinearSVC()
# # lin_clf.fit(xtrain,ytrain1)
# # dec1 = lin_clf.score(xtest,ytest1)
# #print dec1
# # this took a ridiculous amount of time, and only got 49%
#
#
# lassoclf = linear_model.Lasso(alpha = 0.1)
# lassoclf.fit(xtrain,ytrain)
# dec2 = lassoclf.score(xtest,ytest)
# print dec2