import classtest
import FinalClassifier
import AudiovsPower
import NA_Classifier
import AudiovsPower
import scipy.io as sio
import numpy as np

xtrain_aud = sio.loadmat('xtrain_all_aud.mat')
xtrain_aud = xtrain_aud['xtrain']
ybintrain_all_aud = sio.loadmat('ybintrain_all_aud.mat')
ybintrain_all_aud = ybintrain_all_aud['ybintrain']

xtrain_pow = sio.loadmat('xtrain_all_pow.mat')
xtrain_pow = xtrain_pow['xtrain']
ybintrain_all_pow = sio.loadmat('ybintrain_all_pow.mat')
ybintrain_all_pow = ybintrain_all_pow['ybintrain']

xtesting = sio.loadmat('xtesting.mat')
xtesting = xtesting['xtesting']

numfiers = 3
AorPstring = AudiovsPower.myclassify_AudPow(numfiers, xtrain_aud, xtrain_pow, ybintrain_all_aud, ybintrain_all_pow, xtesting)
print AorPstring