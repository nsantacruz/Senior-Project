import NA_Classifier
import scipy.io as sio
import numpy as np
from sklearn.feature_selection import VarianceThreshold as VarThresh


# for feature selection i have a few ideas. 1) run feature selection over the whole matrix of features.
#2) remove some of the recordings and do it a few times (so manually k-folding), because that way if the same features are removed
#then we know that for real those are the features not helpful


xtrain_pow = sio.loadmat('xtrain_all_pow.mat')
xtrain_pow = xtrain_pow['xtrain']
ytrain_pow = sio.loadmat('ytrain_all_pow.mat')
ytrain_pow = ytrain_pow['ytrain']

# method 1: variance threshold

Var_selector = VarThresh(.1)
# without any parameters passed to varthresh it defaults to anything with all feautres the exact same
#  am going to start with .1
Var_selector.fit(xtrain_pow)
which_feats = Var_selector.get_support()
x_pow_fitted = Var_selector.transform(xtrain_pow)