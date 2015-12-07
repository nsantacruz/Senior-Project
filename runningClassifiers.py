
import classtest
import FinalClassifier
import AudiovsPower
import NA_Classifier
import scipy.io as sio
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import ExtraTreeClassifier as ETC
import Transfer_Mat_From_Matlab
import classtest
import FinalClassifier
import AudiovsPower
import NA_Classifier
import scipy.io as sio
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import ExtraTreeClassifier as ETC
import Transfer_Mat_From_Matlab
from Transfer_Mat_From_Matlab import txmat



xtrain_BCDEFGHI = txmat('xtrain_BCDEFGHI_pow.mat','xtrain')
ytrain_BCDEFGHI = txmat('ytrain_BCDEFGHI_pow.mat','ytrain')
xtltrain_BCDEFGHI = txmat('xtltrain_BCDEFGHI_pow.mat','xtltrain')

xtest_BCDEFGHI = txmat('xtest_BCDEFGHI_pow.mat','xtest')
ytest_BCDEFGHI = txmat('ytest_BCDEFGHI_pow.mat','ytest')
xtltest_BCDEFGHI = txmat('xtltest_BCDEFGHI_pow.mat','xtltest')

xtrain_A = txmat('xtrain_A_pow.mat','xtrain') # we're gonna use this for testing'
ytrain_A = txmat('ytrain_A_pow.mat','ytrain')
xtltrain_A = txmat('xtltrain_A_pow.mat','xtltrain')

xtrain_pow = sio.loadmat('xtrain_all_pow.mat')
xtrain_pow = xtrain_pow['xtrain']


print 'NA classifier pow training BCDEFGHI and A'
nu = [0.05, 0.1,.2,.3,.4, 0.5, 0.8]
for param in nu:
    ystring,ystring1 = NA_Classifier.myclassify_NA(2,xtrain_BCDEFGHI,xtest_BCDEFGHI,xtltest_BCDEFGHI,xtrain_A,xtltrain_A,nuparam=param)
    print 'for nu =' + str(param)
    print 'results on BCDEFGHI testing set'
    print ystring
    print 'results on grid A data set'
    print ystring1
    print '/n /n'




xtrain_ABCDEFGH = txmat('xtrain_ABCDEFGH_pow.mat','xtrain')
ytrain_ABCDEFGH = txmat('ytrain_ABCDEFGH_pow.mat','ytrain')
xtltrain_ABCDEFGH = txmat('xtltrain_ABCDEFGH_pow.mat','xtltrain')

xtest_ABCDEFGH = txmat('xtest_ABCDEFGH_pow.mat','xtest')
ytest_ABCDEFGH= txmat('ytest_ABCDEFGHpow.mat','ytest')
xtltest_ABCDEFGH = txmat('xtltest_ABCDEFGH_pow.mat','xtltest')

xtrain_I = txmat('xtrain_I_pow.mat','xtrain') # we're gonna use this for testing'
ytrain_I = txmat('ytrain_I_pow.mat','ytrain')
xtltrain_I = txmat('xtltrain_I_pow.mat','xtltrain')


print 'NA classifier pow training ABCDEFGH and I'
nu = [0.05, 0.1,.2,.3,.4, 0.5, 0.8]
for param in nu:
    ystring,ystring1 = NA_Classifier.myclassify_NA(2,xtrain_ABCDEFGH,xtest_ABCDEFGH,xtltest_ABCDEFGH,xtrain_I,xtltrain_I,nuparam=param)
    print 'for nu =' + str(param)
    print 'results on ABCDEFGH testing set'
    print ystring
    print 'results on grid I data set'
    print ystring1




# ystring0 = NA_Classifier.myclassify_NA(1,xtrain_pow,xtesting)
# print 'NA classifier pow'
# print ystring0


# ystring3 = NA_Classifier.myclassify_NA(1,xtrain_pow_shortened,xtrain_pow_shortened)
# print 'NA classifier pow training'
# print ystring3




#
# xtrain_pow = sio.loadmat('xtrain_all_pow.mat')
# xtrain_pow = xtrain_pow['xtrain']
#
#
# xtrain_pow_shortened = xtrain_pow[:,0:6]
#
# xtrunclength = sio.loadmat('xtrunclength.mat')
# xtrunclength = xtrunclength['xtrunclength']
#
# xtesting = sio.loadmat('xtesting.mat')
# xtesting = xtesting['xtesting']
#
# xtesting = xtesting[~np.isnan(xtesting).any(axis=1),:]
# xtesting = xtesting[~np.isinf(xtesting).any(axis=1),:]
# xtesting_shortened = xtesting[:,0:6]




# ystring3 = NA_Classifier.myclassify_NA(2,xtrain_pow_shortened,xtesting_shortened)
# print 'NA classifier pow testing'
# print ystring3





#yteststring = FinalClassifier.myclassify_practice_set(numfiers=6,xtrain=xtrainall,ytrain=ytrainall1,xtest=xtestall)
#classtest.myclassify(numfiers=5,xtrain=xtrain,ytrain=ytrain1,xtest=xtest,ytest=ytest1)

#print yteststring

# xtrain_aud = sio.loadmat('xtrain_all_aud.mat')
# xtrain_aud = xtrain_aud['xtrain']
#
# xtrain_aud = sio.loadmat('xtrain_all_aud.mat')
# xtrain_aud = xtrain_aud['xtrain']
#
# xtrain_pow = sio.loadmat('xtrain_all_pow.mat')
# xtrain_pow = xtrain_pow['xtrain']
#
# ybin_aud = sio.loadmat('ybintrain_all_aud.mat')
# ybin_aud = ybin_aud['ybintrain']
#
# ybin_pow = sio.loadmat('ybintrain_all_pow.mat')
# ybin_pow = ybin_pow['ybintrain']
#
# xtesting = sio.loadmat('xtesting.mat')
# xtesting = xtesting['xtesting']



# yteststring = AudiovsPower.myclassify_AudPow(3, xtrain_aud, xtrain_pow, ybin_aud, ybin_pow, xtesting)
# print 'binary audio vs power'
# print yteststring



#
# xtrunclength = sio.loadmat('xtrunclength.mat')
# xtrunclength = xtrunclength['xtrunclength']
#
# xtesting = sio.loadmat('xtesting.mat')
# xtesting = xtesting['xtesting']

# xtesting = xtesting[~np.isnan(xtesting).any(axis=1),:]
# xtesting = xtesting[~np.isinf(xtesting).any(axis=1),:]
#
# yteststring = AudiovsPower.myclassify_AudPow(2, xtrain_aud, xtrain_pow, ybin_aud, ybin_pow, xtesting)
# print yteststring[1]
# indsAud = AudiovsPower.indAudPow(yteststring[1], 'A')
# print indsAud
# indsPow = AudiovsPower.indAudPow(yteststring[1], 'P')
# print indsPow

# xtrain_aud = sio.loadmat('xtrain_all_aud.mat')
# xtrain_aud = xtrain_aud['xtrain']
#
# ytrain_aud = sio.loadmat('ytrain_all_aud.mat')
# ytrain_aud = ytrain_aud['ytrain']
#
#
# xtrain_aud_shortened = xtrain_aud[:,0:6]
# xtesting_shortened = xtesting[:,0:6]
#
# xtrain_pow = sio.loadmat('xtrain_all_pow.mat')
# xtrain_pow = xtrain_pow['xtrain']
#
#
# xtrain_pow_shortened = xtrain_pow[:,0:6]
#
#

# ystring0 = NA_Classifier.myclassify_NA(1,xtrain_pow,xtesting)
# print 'NA classifier pow'
# print ystring0

#
# ystring3 = NA_Classifier.myclassify_NA(2,xtrain_pow,xtrain_pow)
# print 'NA classifier pow'
# print ystring3







# ystring2 = NA_Classifier.myclassify_NA(2,xtrain_aud_shortened,xtrain_aud_shortened)
# print 'NA classifier Aud'
# print ystring2
#
# xtrain_pow = sio.loadmat('xtrain_all_pow.mat')
# xtrain_pow = xtrain_pow['xtrain']
#

#xtrain_pow_shortened = xtrain_pow[:,0:6]
#
# ystring1 = NA_Classifier.myclassify_NA(2,xtrain_pow,xtesting)
# print 'NA classifier Pow'
# print ystring1

# aud_class_target_string = FinalClassifier.myclassify_practice_set(numfiers=3,xtrain=xtrain_aud,ytrain=ytrain_aud,xtest=xtesting)
# print 'audio classifier results'
# print aud_class_target_string
#
# xtrain_pow = sio.loadmat('xtrain_all_pow.mat')
# xtrain_pow = xtrain_pow['xtrain']
#
# ytrain_pow = sio.loadmat('ytrain_all_pow.mat')
# ytrain_pow = ytrain_pow['ytrain']
#
#
# pow_class_target_string = FinalClassifier.myclassify_practice_set(numfiers=3,xtrain=xtrain_pow,ytrain=ytrain_pow,xtest=xtesting)
# print 'power classifier results'
# print pow_class_target_string










#
# xtest_audpow = xtest_audpow[~np.isnan(xtest_audpow).any(axis=1),:]
# yteststring = AudiovsPower.myclassify_AudPow(5,xtrain_aud,xtrain_pow,ytrain_aud,ytrain_pow,xtest_audpow)
# print yteststring
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


