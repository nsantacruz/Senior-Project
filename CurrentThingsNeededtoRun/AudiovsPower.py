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


def myclassify_AudPow(numfiers,xtrain_1,xtrain_2,ytrain_1,ytrain_2,xtest):

    # remove NaN, Inf, and -Inf values from the xtest feature matrix
    xtest = xtest[~np.isnan(xtest).any(axis=1),:]
    xtest = xtest[~np.isinf(xtest).any(axis=1),:]

    xtrain = np.append(xtrain_1,xtrain_2,0)
    ytrain = np.append(ytrain_1,ytrain_2)
    ytrain = np.ravel(ytrain)
    xtrunclength = sio.loadmat('Files/xtrunclength.mat')
    xtrunclength = xtrunclength['xtrunclength'][0]



    #if xtest is NxM matrix, returns Nxnumifiers matrix where each column corresponds to a classifiers prediction vector
    count = 0
    # print numfiers

    predictionMat = np.empty((xtest.shape[0],numfiers))
    predictionStringMat = []
    finalPredMat = []

    bagging2 = BaggingClassifier(ETC(),bootstrap=False,bootstrap_features=False)
    bagging2.fit(xtrain,ytrain)
    #print bagging2.score(xtest,ytest)
    ytest = bagging2.predict(xtest)
    predictionMat[:,count] = ytest
    count += 1


    if count < numfiers:

        tree2 = ETC()
        tree2.fit(xtrain,ytrain)
        ytest = tree2.predict(xtest)
        predictionMat[:,count] = ytest
        count+=1


    if count < numfiers:
        bagging1 = BaggingClassifier(ETC())
        bagging1.fit(xtrain,ytrain)
        #print bagging1.score(xtest,ytest)
        ytest = bagging1.predict(xtest)
        predictionMat[:,count] = ytest
        count+=1

    if count < numfiers:
        # votingClassifiers combine completely different machine learning classifiers and use a majority vote
        clff1 = SVC()
        clff2 = RFC(bootstrap=False)
        clff3 = ETC()
        clff4 = neighbors.KNeighborsClassifier()
        clff5 = quadda()



        eclf = VotingClassifier(estimators = [('svc',clff1),('rfc',clff2),('etc',clff3),('knn',clff4),('qda',clff5)])
        eclf = eclf.fit(xtrain,ytrain)
        #print(eclf.score(xtest,ytest))
        # for claf, label in zip([clff1,clff2,clff3,clff4,clff5,eclf],['SVC','RFC','ETC','KNN','QDA','Ensemble']):
        #     cla
        #     scores = crossvalidation.cross_val_score(claf,xtrain,ytrain,scoring='accuracy')
        #     print ()
        ytest = eclf.predict(xtest)
        predictionMat[:,count] = ytest
        count+=1


    if count < numfiers:
        svc1 = SVC()
        svc1.fit(xtrain,ytrain)
        ytest = svc1.predict(xtest)
        predictionMat[:,count] = ytest
        count+=1

    if count < numfiers:
        # Quadradic discriminant analysis - classifier with quadratic decision boundary -
        qda = quadda()
        qda.fit(xtrain,ytrain)
        #print(qda.score(xtest,ytest))
        ytest = qda.predict(xtest)
        predictionMat[:,count] = ytest
        count+=1


    if count < numfiers:

        tree1 = DTC()
        tree1.fit(xtrain,ytrain)
        ytest = tree1.predict(xtest)
        predictionMat[:,count] = ytest
        count+=1

    if count < numfiers:
        knn1 = neighbors.KNeighborsClassifier() # this classifies based on the #k nearest neighbors, where k is definted by the user.
        knn1.fit(xtrain,ytrain)
        ytest = knn1.predict(xtest)
        predictionMat[:,count] = ytest
        count+=1


    if count < numfiers:
        # linear discriminant analysis - classifier with linear decision boundary -
        lda = linda()
        lda.fit(xtrain,ytrain)
        ytest = lda.predict(xtest)
        predictionMat[:,count] = ytest
        count+=1

    if count < numfiers:
        tree3 = RFC()
        tree3.fit(xtrain,ytrain)
        ytest = tree3.predict(xtest)
        predictionMat[:,count] = ytest
        count+=1

    if count < numfiers:
        bagging3 = BaggingClassifier(RFC(),bootstrap=False,bootstrap_features=False)
        bagging3.fit(xtrain,ytrain)
        ytest = bagging3.predict(xtest)
        predictionMat[:,count] = ytest
        count+=1



    if count < numfiers:
        bagging4 = BaggingClassifier(SVC(),bootstrap=False,bootstrap_features=False)
        bagging4.fit(xtrain,ytrain)
        ytest = bagging4.predict(xtest)
        predictionMat[:,count] = ytest
        count+=1


    if count < numfiers:
        tree4 = RFC(bootstrap=False)
        tree4.fit(xtrain,ytrain)
        ytest = tree4.predict(xtest)
        predictionMat[:,count] = ytest
        count+=1

    if count < numfiers:
        tree6 = GBC()
        tree6.fit(xtrain,ytrain)
        ytest = tree6.predict(xtest)
        predictionMat[:,count] = ytest
        count+=1

    if count < numfiers:
        knn2 = neighbors.KNeighborsClassifier(n_neighbors = 10)
        knn2.fit(xtrain,ytrain)
        ytest = knn2.predict(xtest)
        predictionMat[:,count] = ytest
        count+=1


    if count < numfiers:
        knn3 = neighbors.KNeighborsClassifier(n_neighbors = 3)
        knn3.fit(xtrain,ytrain)
        ytest = knn3.predict(xtest)
        predictionMat[:,count] = ytest
        count+=1


    if count < numfiers:
        knn4 = neighbors.KNeighborsClassifier(algorithm = 'ball_tree')
        knn4.fit(xtrain,ytrain)
        ytest = knn4.predict(xtest)
        predictionMat[:,count] = ytest
        count+=1


    if count < numfiers:
        knn5 = neighbors.KNeighborsClassifier(algorithm = 'kd_tree')
        knn5.fit(xtrain,ytrain)
        ytest = knn5.predict(xtest)
        predictionMat[:,count] = ytest
        count+=1


    if count < numfiers:
        ncc1 = NearestCentroid()
        ncc1.fit(xtrain,ytrain)
        ytest = ncc1.predict(xtest)
        predictionMat[:,count] = ytest
        count+=1

    if count < numfiers:
        tree5 = ABC()
        tree5.fit(xtrain,ytrain)
        ytest = tree5.predict(xtest)
        predictionMat[:,count] = ytest
        count+=1



    for colCount in range(predictionMat.shape[1]):
        tempCol = predictionMat[:,colCount]
        modeCol = predWindowVecModeFinder(tempCol,xtrunclength)
        modeStr = predVec2Str(modeCol)
        predictionStringMat.append(modeStr)
        finalPredMat += map(int,modeCol)

    return predictionStringMat,finalPredMat


#given prediction vector for all windows and all recordings, output mode for each recording
def predWindowVecModeFinder(predVec,xtrunclength):
    predModeVec = []
    for count in range(len(xtrunclength)):
        start = 0 if count == 0 else xtrunclength[count-1]
        tempPredRec = predVec[start:xtrunclength[count]]
        from collections import Counter
        b = Counter(tempPredRec)
        predModeVec.append(b.most_common(1)[0][0])
    return predModeVec

def predVec2Str(ytest):
    gridLetters = 'AP'
    str = ''
    for pred in ytest:
        #remember, Audio corresponds to 0
        str+= gridLetters[int(pred)]
    return str

def indAudPow(yteststring, sub):
    # returns indices of yteststring corresponding to sub
    # yteststring is output of AudiovsPower classifier
    # sub is either 'A' or 'P'
    index = 0
    count = 0
    inds = np.empty(yteststring.count(sub))
    while index < len(yteststring):
        index = yteststring.find(sub, index)
        if index == -1:
            break
        inds[count] = index
        count += 1
        index += len(sub)
    return inds

def isRecType(yteststring, sub):
    # returns array of same length as yteststring - returns 1s for match of sub
    # yteststring is output of AudiovsPower classifier
    # sub is either 'A' or 'P'
    index = 0
    isType = np.empty(len(yteststring))
    while index < len(yteststring):
        if yteststring[index] == sub:
            isType[index] = 1
        else:
            isType[index] = 0
        index += 1
    return isType

def splitAudPow(xtesting, xtrunclength, inds):
    for count in range(len(inds)):
        count += 1
        
