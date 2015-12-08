import NA_Classifier
import scipy.io as sio
import numpy as np
from sklearn.feature_selection import VarianceThreshold as VarThresh


# for feature selection i have a few ideas. 1) run feature selection over the whole matrix of features.
#2) remove some of the recordings and do it a few times (so manually k-folding), because that way if the same features are removed
#then we know that for real those are the features not helpful


xtrain_aud = sio.loadmat('xtrain_all_aud.mat')
xtrain_aud = xtrain_aud['xtrain']
ytrain_aud = sio.loadmat('ytrain_all_aud.mat')
ytrain_aud = ytrain_aud['ytrain']

# method 1: variance threshold

Var_selector = VarThresh(.5)
# without any parameters passed to varthresh it defaults to anything with all feautres the exact same
#  am going to start with .1
Var_selector.fit(xtrain_aud)
which_feats = Var_selector.get_support()
x_aud_fitted = Var_selector.transform(xtrain_aud)

print x_aud_fitted.shape


xtrunclength = sio.loadmat('xtrunclength.mat')
xtrunclength = xtrunclength['xtrunclength']

xtesting = sio.loadmat('xtesting.mat')
xtesting = xtesting['xtesting']

xtesting = xtesting[~np.isnan(xtesting).any(axis=1),:]
xtesting = xtesting[~np.isinf(xtesting).any(axis=1),:]






import FinalClassifier
xtesting = sio.loadmat('xtesting.mat')
xtesting = xtesting['xtesting']

xtesting = xtesting[~np.isnan(xtesting).any(axis=1),:]
xtesting = xtesting[~np.isinf(xtesting).any(axis=1),:]
xtesting_new = xtesting[:,which_feats]
#xtesting_new = xtesting[:,0:1]
#x_aud_fitted = xtrain_aud[:,0:1]
print x_aud_fitted.shape
print xtesting_new.shape
#xtesting_shortened = xtesting[:,0:6]
print 'getting ready to test'
pow_class_target_string = FinalClassifier.myclassify_practice_set(numfiers=3,xtrain=xtrain_aud,ytrain=ytrain_aud,xtest=xtesting)
print 'power classifier results'
print pow_class_target_string


pow_class_target_string1 = FinalClassifier.myclassify_practice_set(numfiers=3,xtrain=x_aud_fitted,ytrain=ytrain_aud,xtest=xtesting_new)
print 'power classifier results'
print pow_class_target_string1