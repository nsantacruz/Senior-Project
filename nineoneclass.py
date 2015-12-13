import scipy.io as sio
from Transfer_Mat_From_Matlab import txmat
import numpy as np
from sklearn.svm import OneClassSVM as oneclass
import NA_Classifier


#Grid A
xtrain_A_pow = txmat('xtrain_A_pow.mat','xtrain')
xtltrain_A_pow = txmat('xtltrain_A_pow.mat','xtltrain')
#Grid B
xtrain_B_pow = txmat('xtrain_B_pow.mat','xtrain')
xtltrain_B_pow = txmat('xtltrain_B_pow.mat','xtltrain')
#Grid C
xtrain_C_pow = txmat('xtrain_C_pow.mat','xtrain')
xtltrain_C_pow = txmat('xtltrain_C_pow.mat','xtltrain')

xtrain_D_pow = txmat('xtrain_D_pow.mat','xtrain')
xtltrain_D_pow = txmat('xtltrain_D_pow.mat','xtltrain')

xtrain_E_pow = txmat('xtrain_E_pow.mat','xtrain')
xtltrain_E_pow = txmat('xtltrain_E_pow.mat','xtltrain')

xtrain_F_pow = txmat('xtrain_F_pow.mat','xtrain')
xtltrain_F_pow = txmat('xtltrain_F_pow.mat','xtltrain')

xtrain_G_pow = txmat('xtrain_G_pow.mat','xtrain')
xtltrain_G_pow = txmat('xtltrain_G_pow.mat','xtltrain')

xtrain_H_pow = txmat('xtrain_H_pow.mat','xtrain')
xtltrain_H_pow = txmat('xtltrain_H_pow.mat','xtltrain')

xtrain_I_pow = txmat('xtrain_I_pow.mat','xtrain')
xtltrain_I_pow = txmat('xtltrain_I_pow.mat','xtltrain')






ystring = NA_Classifier.myclassify_oneclass(1,xtrain_A_pow,xtrain_A_pow,xtltrain_A_pow,nuparam = .3)
print 'results for train A, nu = .3'
print ""
print ystring
ystring = NA_Classifier.myclassify_oneclass(1,xtrain_A_pow,xtrain_B_pow,xtltrain_B_pow,nuparam = .3)
print 'results on B'
print ystring
ystring = NA_Classifier.myclassify_oneclass(1,xtrain_A_pow,xtrain_C_pow,xtltrain_C_pow,nuparam = .3)
print 'results on C'
print ystring
ystring = NA_Classifier.myclassify_oneclass(1,xtrain_A_pow,xtrain_D_pow,xtltrain_D_pow,nuparam = .3)
print 'results on D'
print ystring
ystring = NA_Classifier.myclassify_oneclass(1,xtrain_A_pow,xtrain_E_pow,xtltrain_E_pow,nuparam = .3)
print 'results on E'
print ystring
ystring = NA_Classifier.myclassify_oneclass(1,xtrain_A_pow,xtrain_F_pow,xtltrain_F_pow,nuparam = .3)
print 'results on F'
print ystring
ystring = NA_Classifier.myclassify_oneclass(1,xtrain_A_pow,xtrain_G_pow,xtltrain_G_pow,nuparam = .3)
print 'results on G'
print ystring
ystring = NA_Classifier.myclassify_oneclass(1,xtrain_A_pow,xtrain_H_pow,xtltrain_H_pow,nuparam = .3)
print 'results on H'
print ystring
ystring = NA_Classifier.myclassify_oneclass(1,xtrain_A_pow,xtrain_I_pow,xtltrain_I_pow,nuparam = .3)
print 'results on I'
print ystring







print 'now for nu = .2'
ystring = NA_Classifier.myclassify_oneclass(1,xtrain_A_pow,xtrain_A_pow,xtltrain_A_pow,nuparam = .2)
print 'results for train A, nu = .2'
print ""
print ystring
ystring = NA_Classifier.myclassify_oneclass(1,xtrain_A_pow,xtrain_B_pow,xtltrain_B_pow,nuparam = .2)
print 'results on B'
print ystring
ystring = NA_Classifier.myclassify_oneclass(1,xtrain_A_pow,xtrain_C_pow,xtltrain_C_pow,nuparam = .2)
print 'results on C'
print ystring
ystring = NA_Classifier.myclassify_oneclass(1,xtrain_A_pow,xtrain_D_pow,xtltrain_D_pow,nuparam = .2)
print 'results on D'
print ystring
ystring = NA_Classifier.myclassify_oneclass(1,xtrain_A_pow,xtrain_E_pow,xtltrain_E_pow,nuparam = .2)
print 'results on E'
print ystring
ystring = NA_Classifier.myclassify_oneclass(1,xtrain_A_pow,xtrain_F_pow,xtltrain_F_pow,nuparam = .2)
print 'results on F'
print ystring
ystring = NA_Classifier.myclassify_oneclass(1,xtrain_A_pow,xtrain_G_pow,xtltrain_G_pow,nuparam = .2)
print 'results on G'
print ystring
ystring = NA_Classifier.myclassify_oneclass(1,xtrain_A_pow,xtrain_H_pow,xtltrain_H_pow,nuparam = .2)
print 'results on H'
print ystring
ystring = NA_Classifier.myclassify_oneclass(1,xtrain_A_pow,xtrain_I_pow,xtltrain_I_pow,nuparam = .2)
print 'results on I'
print ystring





# tried nu = .7, it didnt work. nu = .5 got about 50% right for A, but 100% right on other grids.
# nu = .3 worked well, going to try nu = .2 soon




