from sklearn.grid_search import GridSearchCV
from sklearn.svm import OneClassSVM as oneclass
from sklearn.metrics import recall_score as recsco
import scipy.io as sio
import numpy as np

xtrunclength = sio.loadmat('xtrunclength.mat')
xtrunclength = xtrunclength['xtrunclength']

xtesting = sio.loadmat('xtesting.mat')
xtesting = xtesting['xtesting']

xtrain_aud = sio.loadmat('xtrain_all_aud.mat')
xtrain_aud = xtrain_aud['xtrain']

ytrain_aud = sio.loadmat('ytrain_all_aud.mat')
ytrain_aud = ytrain_aud['ytrain']


xtrain_aud_shortened = xtrain_aud[:,0:6]
xtesting_shortened = xtesting[:,0:6]

xtrain_pow = sio.loadmat('xtrain_all_pow.mat')
xtrain_pow = xtrain_pow['xtrain']


xtrain_pow_shortened = xtrain_pow[:,0:6]








tuned_params = [{'kernel': ['rbf'],'tol' : [.9,1]}] #,{'kernel': ['linear']},{'kernel': ['poly'], 'degree': [2,6],'gamma':[.1,.0001],'coef0': [0.0,1.0]}]
#indep_grid ={'iid': [True,False]} then use ParameterGrid?


oc = oneclass()
oc.fit(xtrain_aud_shortened)
predVec = oc.predict(xtrain_aud_shortened)
predVec = (predVec+1)/2
for i in range(len(predVec)):
    predVec[i] = int(predVec[i])

score = recsco(np.ones(xtrain_aud_shortened.shape),predVec)
print "Score %s" % score


#gridSearch = GridSearchCV(estimator = oneclass(),param_grid = tuned_params,iid=True,scoring=recsco)
print "starting grid search"
#gridSearch.fit(xtrain_aud)
print("Best params found")
# print()
# print(gridSearch.best_params_)
# print("Grid scores on development set")
# print()
