import scipy.io as sio
import NA_Classifier
from Transfer_Mat_From_Matlab import txmat
import FinalClassifier




#-----first data set

xtrain_1test_aud = txmat('xtrain_1test_aud.mat','xtrain')
ytrain_1test_aud = txmat('ytrain_1test_aud.mat','ytrain')
xtltrain_1test_aud = txmat('xtltrain_1test_aud.mat','xtltrain')

xtest_1test_aud = txmat('xtest_1test_aud.mat','xtest')
ytest_1test_aud = txmat('ytest_1test_aud.mat','ytest')
xtltest_1test_aud = txmat('xtltest_1test_aud.mat','xtltest')

ystring,ytarg = FinalClassifier.myclassify_practice_set(1,xtrain_1test_aud,ytrain_1test_aud,xtltrain_1test_aud,xtltest_1test_aud,xtest_1test_aud,ytest_1test_aud)

print format('Target: ',"12s"), ytarg
print format('Predictions: ',"11s"), ystring



xtrain_BD_aud = txmat('xtrain_BD_aud.mat','xtrain')
ytrain_BD_aud = txmat('ytrain_BD_aud.mat','ytrain')
xtltrain_BD_aud = txmat('xtltrain_BD_aud.mat','xtltrain')


ystringBD,ytargBD = FinalClassifier.myclassify_practice_set(1,xtrain_BD_aud,ytrain_BD_aud,xtltrain_BD_aud,xtltest_1test_aud,xtest_1test_aud,ytest_1test_aud,testing=True,grids = 'BD')
print format('Target: ',"12s"), ytargBD
print format('Predictions: ',"11s"), ystringBD

xtrain_BE_aud = txmat('xtrain_BE_aud.mat','xtrain')
ytrain_BE_aud = txmat('ytrain_BE_aud.mat','ytrain')
xtltrain_BE_aud = txmat('xtltrain_BE_aud.mat','xtltrain')

ystringBE,ytargBE = FinalClassifier.myclassify_practice_set(1,xtrain_BE_aud,ytrain_BE_aud,xtltrain_BE_aud,xtltest_1test_aud,xtest_1test_aud,ytest_1test_aud,testing=True,grids='BE')
print format('Target: ',"12s"), ytargBE
print format('Predictions: ',"11s"), ystringBE







#----second data set
print("")
print("")
print("Second Data Set")



xtrain_1test_aud = txmat('xtrain_1test2_aud.mat','xtrain')
ytrain_1test_aud = txmat('ytrain_1test2_aud.mat','ytrain')
xtltrain_1test_aud = txmat('xtltrain_1test2_aud.mat','xtltrain')

xtest_1test_aud = txmat('xtest_1test2_aud.mat','xtest')
ytest_1test_aud = txmat('ytest_1test2_aud.mat','ytest')
xtltest_1test_aud = txmat('xtltest_1test2_aud.mat','xtltest')

ystring,ytarg= FinalClassifier.myclassify_practice_set(1,xtrain_1test_aud,ytrain_1test_aud,xtltrain_1test_aud,xtltest_1test_aud,xtest_1test_aud,ytest_1test_aud)

print format('Target: ',"12s"), ytarg
print format('Predictions: ',"11s"), ystring


xtrain_BD_aud = txmat('xtrain_BD_aud.mat','xtrain')
ytrain_BD_aud = txmat('ytrain_BD_aud.mat','ytrain')
xtltrain_BD_aud = txmat('xtltrain_BD_aud.mat','xtltrain')


ystringBD,ytargBD = FinalClassifier.myclassify_practice_set(1,xtrain_BD_aud,ytrain_BD_aud,xtltrain_BD_aud,xtltest_1test_aud,xtest_1test_aud,ytest_1test_aud,testing=True,grids = 'BD')
print format('Target: ',"12s"), ytargBD
print format('Predictions: ',"11s"), ystringBD

xtrain_BE_aud = txmat('xtrain_BE_aud.mat','xtrain')
ytrain_BE_aud = txmat('ytrain_BE_aud.mat','ytrain')
xtltrain_BE_aud = txmat('xtltrain_BE_aud.mat','xtltrain')

ystringBE,ytargBE = FinalClassifier.myclassify_practice_set(1,xtrain_BE_aud,ytrain_BE_aud,xtltrain_BE_aud,xtltest_1test_aud,xtest_1test_aud,ytest_1test_aud,testing=True,grids='BE')
print format('Target: ',"12s"), ytargBE
print format('Predictions: ',"11s"), ystringBE




#-----third data set

print("")
print("")
print("Third Data Set")



xtrain_1test_aud = txmat('xtrain_1test3_aud.mat','xtrain')
ytrain_1test_aud = txmat('ytrain_1test3_aud.mat','ytrain')
xtltrain_1test_aud = txmat('xtltrain_1test3_aud.mat','xtltrain')

xtest_1test_aud = txmat('xtest_1test3_aud.mat','xtest')
ytest_1test_aud = txmat('ytest_1test3_aud.mat','ytest')
xtltest_1test_aud = txmat('xtltest_1test3_aud.mat','xtltest')

ystring,ytarg = FinalClassifier.myclassify_practice_set(1,xtrain_1test_aud,ytrain_1test_aud,xtltrain_1test_aud,xtltest_1test_aud,xtest_1test_aud,ytest_1test_aud)

print format('Target: ',"12s"), ytarg
print format('Predictions: ',"11s"), ystring

xtrain_BD_aud = txmat('xtrain_BD_aud.mat','xtrain')
ytrain_BD_aud = txmat('ytrain_BD_aud.mat','ytrain')
xtltrain_BD_aud = txmat('xtltrain_BD_aud.mat','xtltrain')


ystringBD,ytargBD = FinalClassifier.myclassify_practice_set(1,xtrain_BD_aud,ytrain_BD_aud,xtltrain_BD_aud,xtltest_1test_aud,xtest_1test_aud,ytest_1test_aud,testing=True,grids = 'BD')
print format('Target: ',"12s"), ytargBD
print format('Predictions: ',"11s"), ystringBD

xtrain_BE_aud = txmat('xtrain_BE_aud.mat','xtrain')
ytrain_BE_aud = txmat('ytrain_BE_aud.mat','ytrain')
xtltrain_BE_aud = txmat('xtltrain_BE_aud.mat','xtltrain')

ystringBE,ytargBE = FinalClassifier.myclassify_practice_set(1,xtrain_BE_aud,ytrain_BE_aud,xtltrain_BE_aud,xtltest_1test_aud,xtest_1test_aud,ytest_1test_aud,testing=True,grids='BE')
print format('Target: ',"12s"), ytargBE
print format('Predictions: ',"11s"), ystringBE




# Now to test binary SVM's!
#
#
# xtrain_BCDEFGHI = txmat('xtrain_BCDEFGHI_pow.mat','xtrain')
# ytrain_BCDEFGHI = txmat('ytrain_BCDEFGHI_pow.mat','ytrain')
# xtltrain_BCDEFGHI = txmat('xtltrain_BCDEFGHI_pow.mat','xtltrain')
#
# xtest_BCDEFGHI = txmat('xtest_BCDEFGHI_pow.mat','xtest')
# ytest_BCDEFGHI = txmat('ytest_BCDEFGHI_pow.mat','ytest')
# xtltest_BCDEFGHI = txmat('xtltest_BCDEFGHI_pow.mat','xtltest')
#
# xtrain_A = txmat('xtrain_A_pow.mat','xtrain') # we're gonna use this for testing'
# ytrain_A = txmat('ytrain_A_pow.mat','ytrain')
# xtltrain_A = txmat('xtltrain_A_pow.mat','xtltrain')
#
# ystring = FinalClassifier.myclassify_practice_set(1,xtrain_BCDEFGHI,ytrain_BCDEFGHI,xtltrain_BCDEFGHI,xtltest_BCDEFGHI,xtest_BCDEFGHI,ytest_BCDEFGHI)
#
# print ystring




#
# xtrain_ABCDEFGH = txmat('xtrain_ABCDEFGH_pow.mat','xtrain')
# ytrain_ABCDEFGH = txmat('ytrain_ABCDEFGH_pow.mat','ytrain')
# xtltrain_ABCDEFGH = txmat('xtltrain_ABCDEFGH_pow.mat','xtltrain')
#
# xtest_ABCDEFGH = txmat('xtest_ABCDEFGH_pow.mat','xtest')
# ytest_ABCDEFGH= txmat('ytest_ABCDEFGH_pow.mat','ytest')
# xtltest_ABCDEFGH = txmat('xtltest_ABCDEFGH_pow.mat','xtltest')
#
# xtrain_I = txmat('xtrain_I_pow.mat','xtrain') # we're gonna use this for testing'
# ytrain_I = txmat('ytrain_I_pow.mat','ytrain')
# xtltrain_I = txmat('xtltrain_I_pow.mat','xtltrain')
#
# ystring = FinalClassifier.myclassify_practice_set(1,xtrain_ABCDEFGH,ytrain_ABCDEFGH,xtltrain_ABCDEFGH,xtltest_ABCDEFGH,xtest_ABCDEFGH,ytest_ABCDEFGH)
# print ystring





