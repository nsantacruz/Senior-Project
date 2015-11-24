
import classtest
import scipy.io as sio
import numpy as np

xtrain=sio.loadmat('xtrainR1.mat')
# print xtrain

xtrain=xtrain['xtrain']
ytrain=sio.loadmat('ytrainR1.mat')
ytrain = ytrain['ytrain']
xtest=sio.loadmat('xtestR1.mat')
xtest = xtest['xtest']
ytest=sio.loadmat('ytestR1.mat')
ytest=ytest['ytest']

ytrain1 = np.ravel(ytrain)
ytest1 = np.ravel(ytest)



classtest.myclassify(numfiers=21,xtrain=xtrain,ytrain=ytrain1,xtest=xtest,ytest=ytest1)

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