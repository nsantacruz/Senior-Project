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
from sklearn.metrics import confusion_matrix
#def myclassify(numfiers = 6,xtrain=xtrain,ytrain=ytrain,xtest=xtest,ytest=ytest):
def myclassify_practice_set(numfiers,xtrain,ytrain,xtltrain,xtltest,xtest,ytarget=None,testing=False,grids='ABCDEFGHI'):
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
    # print numfiers

    predictionMat = np.empty((xtest.shape[0],numfiers))
    predictionStringMat = []
    finalPredMat = []
    targetStringMat = []
    targets1 = []
    predictions1 = []

    # svc1 = SVC()
    # svc1.fit(xtrain,ytrain)
    # ytest = svc1.predict(xtest)
    # predictionMat[:,count] = ytest
    # count+=1
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


    # print xtltest
    # print len(ytest)
    for colCount in range(predictionMat.shape[1]):
        tempCol = predictionMat[:,colCount]
        if testing:
            modeCol = temppredWindowVecModeFinder(tempCol,xtltest,4,grids,isPrint=0)
        else:
            modeCol = predWindowVecModeFinder(tempCol,xtltest,4,isPrint=0)

        ytarg = predWindowVecModeFinder(ytarget,xtltest,1,isPrint=0)
        if testing:
             modeStr = temppredVec2Str(modeCol,grids)
        else:
            modeStr = predVec2Str(modeCol)
        modeStrans = predVec2Str(ytarg)
        predictionStringMat.append(modeStr)
        predictions1.append(modeCol)
        finalPredMat += map(int,modeCol)
        targetStringMat.append(modeStrans)
        targets1.append(ytarg)
        if testing == False:
            if ytarget != None:
                #print targets1
                #print ""
                #print predictions1
                confusionme = confusion_matrix(targets1[0],predictions1[0])
                #print "Confusion Matrix is: "
                #print confusionme


    return predictionStringMat, targetStringMat, finalPredMat


#given prediction vector for all windows and all recordings, output mode for each recording
def predWindowVecModeFinder(predVec,xtrunclength,kmost,isPrint):
    kmostcommon = kmost
    predMat = np.zeros([len(xtrunclength),kmostcommon])-1
    percMat = np.zeros(predMat.shape)

    for count in range(len(xtrunclength)):
        start = 0 if count == 0 else xtrunclength[count-1]
        tempPredRec = predVec[start:xtrunclength[count]]
        from collections import Counter
        b = Counter(tempPredRec)
        num_Guesses = len(b.most_common(kmostcommon))

        for guess in range(num_Guesses):
            which_grid = b.most_common()[guess][0]
            how_many = b.most_common()[guess][1]
            predMat[count,guess] = which_grid
            percMat[count,guess] = float(how_many)/len(tempPredRec)
    if isPrint == 1:
        for i in range(kmostcommon):
            print "Mode Classes  " + str(i) + ": " + predVec2Str(predMat[:,i].tolist())
            tempPercList = percMat[:,i].tolist()
            tempPercList = map('{:.3f}'.format,tempPercList)
            print "Mode Percents " + str(i) + ": " + str(tempPercList)
    return predMat[:,0].tolist()




def temppredWindowVecModeFinder(predVec,xtrunclength,kmost,grids,isPrint):
    kmostcommon = kmost
    predMat = np.zeros([len(xtrunclength),kmostcommon])-1
    percMat = np.zeros(predMat.shape)

    for count in range(len(xtrunclength)):
        start = 0 if count == 0 else xtrunclength[count-1]
        tempPredRec = predVec[start:xtrunclength[count]]
        from collections import Counter
        b = Counter(tempPredRec)
        num_Guesses = len(b.most_common(kmostcommon))

        for guess in range(num_Guesses):
            which_grid = b.most_common()[guess][0]
            how_many = b.most_common()[guess][1]
            predMat[count,guess] = which_grid
            percMat[count,guess] = float(how_many)/len(tempPredRec)
    if isPrint == 1:
        for i in range(kmostcommon):
            print "Mode Classes  " + str(i) + ": " + temppredVec2Str(predMat[:,i].tolist(),grids)
            tempPercList = percMat[:,i].tolist()
            tempPercList = map('{:.3f}'.format,tempPercList)
            print "Mode Percents " + str(i) + ": " + str(tempPercList)
    return predMat[:,0].tolist()


#TemporaryFunciton for me to do some testing
def temppredVec2Str(ytest,grids):
    gridLetters = grids
    str = '  '
    if ytest != []:
        for pred in ytest:
            #remember, A corresponds to class 1
            tempInt = int(pred)
            str = str + gridLetters[int(pred)-1] if tempInt != -1 else str + '-'
            str += ' ' * 8
    return str


def predVec2Str(ytest):
    gridLetters = 'ABCDEFGHIN'
    str = ' ' *3
    if ytest != []:
        for pred in ytest:
            #remember, A corresponds to class 1
            tempInt = int(pred)
            str = str + gridLetters[int(pred)-1] if tempInt != -1 else str + '-'
            # str += ' '
    return str

#removes nan and inf rows from mat, while updating xtrunclength to remain in-sync
def removeNanAndInf(mat,xtrunclength,ymat=None):
    badinds = np.isinf(mat).any(axis=1) | np.isnan(mat).any(axis=1)
    # print badinds
    mat = mat[~badinds,:]
    if ymat==None:
        ymat = []
    else:
        ymat = ymat[~badinds,:]
    # ymat = ymat[~badinds,:]
    # print xtrunclength[0]
    xtrunclength = xtrunclength[0]
    # print range(len(xtrunclength))
    for i in range(len(xtrunclength)):
        tempBadInds = badinds[0:xtrunclength[i]]
        xtrunclength[i] = xtrunclength[i] - np.sum(tempBadInds)




    return mat,xtrunclength,ymat
