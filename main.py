import numpy as np
import scipy.io as sio

import NA_Classifier

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
xtesting_shortened = xtesting[:,0:6]

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


ystring3 = NA_Classifier.myclassify_NA(2,xtrain_pow_shortened,xtrain_pow_shortened)
print 'NA classifier pow training'
print ystring3


ystring3 = NA_Classifier.myclassify_NA(2,xtrain_pow_shortened,xtesting_shortened)
print 'NA classifier pow testing'
print ystring3