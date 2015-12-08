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
def myclassify_practice_set(numfiers,xtrain,ytrain,xtest):

    # remove NaN, Inf, and -Inf values from the xtest feature matrix
    xtest = xtest[~np.isnan(xtest).any(axis=1),:]
    xtest = xtest[~np.isinf(xtest).any(axis=1),:]

    ytrain = np.ravel(ytrain)

    xtrunclength = sio.loadmat('xtrunclength.mat')
    xtrunclength = xtrunclength['xtrunclength'][0]

    #if xtest is NxM matrix, returns Nxnumifiers matrix where each column corresponds to a classifiers prediction vector
    count = 0
    print numfiers

    predictionMat = np.empty((xtest.shape[0],numfiers))
    predictionStringMat = []

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
        modeCol,p1,modeCol2,p2,modeCol3,p3,modeCol4,p4 = predWindowVecModeFinder(tempCol,xtrunclength)
        modeStr = predVec2Str(modeCol)
        modeStr2 = predVec2Str(modeCol2)
        modeStr3 = predVec2Str(modeCol3)
        modeStr4 = predVec2Str(modeCol4)
        predictionStringMat.append(modeStr)

    return predictionStringMat


#given prediction vector for all windows and all recordings, output mode for each recording
def predWindowVecModeFinder(predVec,xtrunclength):
    predModeVec = []
    perc = []
    predModeVec1 = []
    perc1 = []
    predModeVec2 = []
    perc2 = []
    predModeVec3 = []
    perc3 = []
    predModeVec4 = []
    perc4 = []
    for count in range(len(xtrunclength)):
        start = 0 if count == 0 else xtrunclength[count-1]
        tempPredRec = predVec[start:xtrunclength[count]]
        from collections import Counter
        b = Counter(tempPredRec)
        predModeVec.append(b.most_common(1)[0][0])
        num_Guesses = len(b.most_common())
        for guess in range(num_Guesses):
            which_grid = b.most_common()[guess][0]
            how_many = b.most_common()[guess][1]
            if guess == 0:
                predModeVec.append(which_grid)
                perc.append(how_many)
            elif guess ==1:
                predModeVec1.append(which_grid)
                perc1.append(how_many)
            elif guess ==2:
                predModeVec2.append(which_grid)
                perc2.append(how_many)
            elif guess ==3:
                predModeVec3.append(which_grid)
                perc3.append(how_many)
            else:
                predModeVec4.append(which_grid)
                perc4.append(how_many)

    return predModeVec,perc,predModeVec1,perc1,predModeVec2,perc2,predModeVec3,perc3

def predVec2Str(ytest):
    gridLetters = 'ABCDEFGHI'
    str = ''
    for pred in ytest:
        #remember, A corresponds to class 1
        str += gridLetters[int(pred)-1]
    return str

