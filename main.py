import classtest
import FinalClassifier
import AudiovsPower
import NA_Classifier
import scipy.io as sio
import numpy as np
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import ExtraTreeClassifier as ETC
from Transfer_Mat_From_Matlab import txmat



#
# xtrain_aud = sio.loadmat('xtrain_all_aud.mat')
# xtrain_aud = xtrain_aud['xtrain']
# #


# yteststring = AudiovsPower.myclassify_AudPow(3, xtrain_aud, xtrain_pow, ybin_aud, ybin_pow, xtesting)
# print 'binary audio vs power'
# print yteststring


#
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
#xtesting_shortened = xtesting[:,0:6]

xtrain_pow = sio.loadmat('xtrain_all_pow.mat')
xtrain_pow = xtrain_pow['xtrain']


xtrain_pow_shortened = xtrain_pow[:,0:6]

xtrunclength = sio.loadmat('xtrunclength.mat')
xtrunclength = xtrunclength['xtrunclength']

xtesting = sio.loadmat('xtesting.mat')
xtesting = xtesting['xtesting']

xtesting = xtesting[~np.isnan(xtesting).any(axis=1),:]
xtesting = xtesting[~np.isinf(xtesting).any(axis=1),:]

# ystring0 = NA_Classifier.myclassify_NA(1,xtrain_pow,xtesting)
# print 'NA classifier pow'
# print ystring0
xtrain_ABCDEFGH = txmat('xtrain_ABCDEFGH_pow.mat','xtrain')
ytrain_ABCDEFGH = txmat('ytrain_ABCDEFGH_pow.mat','ytrain')

ystring = FinalClassifier.myclassify_practice_set(1,xtrain_ABCDEFGH,ytrain_ABCDEFGH,xtesting)