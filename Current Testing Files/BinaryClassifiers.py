import scipy.io as sio
import numpy as np

import numpy as np

#from sklearn import svm
import scipy.io as sio
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression as logreg
from sklearn.feature_selection import SelectFromModel as sfm
from sklearn import neighbors
from sklearn.neighbors.nearest_centroid import NearestCentroid
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as linda
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis as quadda
from sklearn.tree import DecisionTreeClassifier as DTC
from sklearn.tree import ExtraTreeClassifier as ETC
from sklearn.ensemble import BaggingClassifier
from sklearn.ensemble import RandomForestClassifier as RFC
from sklearn.ensemble import AdaBoostClassifier as ABC
from sklearn.ensemble import GradientBoostingClassifier as GBC
from sklearn.ensemble import VotingClassifier

#def myclassify(numfiers = 6,xtrain=xtrain,ytrain=ytrain,xtest=xtest,ytest=ytest):
def myclassify_binary(numfiers,xtrain,ytrain,xtltrain,xtltest,xtest,ytarget=None):
    #NOTE we might not need xtltrain
    # xtrain and ytrain are your training set. xtltrain is the indices of corresponding recordings in xtrain and ytrain. these will always be present
    #xtest is your testing set. xtltest is the corresponding indices of the recording. for the practice set xtltest = xtrunclength
    # ytest is optional and depends on if you are using a testing set or the practice set

    # remove NaN, Inf, and -Inf values from the xtest feature matrix
    xtest,xtltest,ytarget = removeNanAndInf(xtest,xtltest,ytarget)
    # print 'finished removal of Nans'

    ytrain = np.ravel(ytrain)
    ytarget = np.ravel(ytarget)


    #if xtest is NxM matrix, returns Nxnumifiers matrix where each column corresponds to a classifiers prediction vector
    count = 0
    print numfiers

    predictionMat = np.empty((xtest.shape[0],numfiers))
    predictionStringMat = []
    targetStringMat = []

    # svc1 = SVC()
    # svc1.fit(xtrain,ytrain)
    # ytest = svc1.predict(xtest)
    # predictionMat[:,count] = ytest
    # count+=1
