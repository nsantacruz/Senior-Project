

import numpy as np
 


#from sklearn import svm
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

    #if xtest is NxM matrix, returns Nxnumifiers matrix where each column corresponds to a classifiers prediction vector
    count = 0
    print numfiers

    predictionMat = np.empty((xtest.shape[0],numfiers))
    predictionStringMat = np.empty((1,numfiers))

    bagging2 = BaggingClassifier(ETC(),bootstrap=False,bootstrap_features=False)
    bagging2.fit(xtrain,ytrain)
    #print bagging2.score(xtest,ytest)
    ytest = bagging2.predict(xtest)
    predictionMat[:,count] = ytest
    predictionStringMat[count] = predVec2Str(ytest)
    count += 1


    if count < numfiers:

        tree2 = ETC()
        tree2.fit(xtrain,ytrain)
        ytest = tree2.predict(xtest)
        predictionMat[:,count] = ytest
        predictionStringMat[1,count] = predVec2Str(ytest)
        count+=1


    if count < numfiers:
        bagging1 = BaggingClassifier(ETC())
        bagging1.fit(xtrain,ytrain)
        #print bagging1.score(xtest,ytest)
        ytest = bagging1.predict(xtest)
        predictionMat[:,count] = ytest
        predictionStringMat[1,count] = predVec2Str(ytest)
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
        predictionStringMat[1,count] = predVec2Str(ytest)
        count+=1


    if count < numfiers:
        svc1 = SVC()
        svc1.fit(xtrain,ytrain)
        ytest = svc1.predict(xtest)
        predictionMat[:,count] = ytest
        predictionStringMat[1,count] = predVec2Str(ytest)
        count+=1

    if count < numfiers:
        # Quadradic discriminant analysis - classifier with quadratic decision boundary -
        qda = quadda()
        qda.fit(xtrain,ytrain)
        #print(qda.score(xtest,ytest))
        ytest = qda.predict(xtest)
        predictionMat[:,count] = ytest
        predictionStringMat[1,count] = predVec2Str(ytest)
        count+=1



    if count < numfiers:

        tree1 = DTC()
        tree1.fit(xtrain,ytrain)
        ytest = tree1.predict(xtest)
        predictionMat[:,count] = ytest
        predictionStringMat[1,count] = predVec2Str(ytest)
        count+=1

    if count < numfiers:
        knn1 = neighbors.KNeighborsClassifier() # this classifies based on the #k nearest neighbors, where k is definted by the user.
        knn1.fit(xtrain,ytrain)
        ytest = knn1.predict(xtest)
        predictionMat[:,count] = ytest
        predictionStringMat[1,count] = predVec2Str(ytest)
        count+=1


    if count < numfiers:
        # linear discriminant analysis - classifier with linear decision boundary -
        lda = linda()
        lda.fit(xtrain,ytrain)
        ytest = lda.predict(xtest)
        predictionMat[:,count] = ytest
        predictionStringMat[1,count] = predVec2Str(ytest)
        count+=1

    if count < numfiers:
        tree3 = RFC()
        tree3.fit(xtrain,ytrain)
        ytest = tree3.predict(xtest)
        predictionMat[:,count] = ytest
        predictionStringMat[1,count] = predVec2Str(ytest)
        count+=1

    if count < numfiers:
        bagging3 = BaggingClassifier(RFC(),bootstrap=False,bootstrap_features=False)
        bagging3.fit(xtrain,ytrain)
        ytest = bagging3.predict(xtest)
        predictionMat[:,count] = ytest
        predictionStringMat[1,count] = predVec2Str(ytest)
        count+=1



    if count < numfiers:
        bagging4 = BaggingClassifier(SVC(),bootstrap=False,bootstrap_features=False)
        bagging4.fit(xtrain,ytrain)
        ytest = bagging4.predict(xtest)
        predictionMat[:,count] = ytest
        predictionStringMat[1,count] = predVec2Str(ytest)
        count+=1


    if count < numfiers:
        tree4 = RFC(bootstrap=False)
        tree4.fit(xtrain,ytrain)
        ytest = tree4.predict(xtest)
        predictionMat[:,count] = ytest
        predictionStringMat[1,count] = predVec2Str(ytest)
        count+=1

    if count < numfiers:
        tree6 = GBC()
        tree6.fit(xtrain,ytrain)
        ytest = tree6.predict(xtest)
        predictionMat[:,count] = ytest
        predictionStringMat[1,count] = predVec2Str(ytest)
        count+=1

    if count < numfiers:
        knn2 = neighbors.KNeighborsClassifier(n_neighbors = 10)
        knn2.fit(xtrain,ytrain)
        ytest = knn2.predict(xtest)
        predictionMat[:,count] = ytest
        predictionStringMat[1,count] = predVec2Str(ytest)
        count+=1


    if count < numfiers:
        knn3 = neighbors.KNeighborsClassifier(n_neighbors = 3)
        knn3.fit(xtrain,ytrain)
        ytest = knn3.predict(xtest)
        predictionMat[:,count] = ytest
        predictionStringMat[1,count] = predVec2Str(ytest)
        count+=1


    if count < numfiers:
        knn4 = neighbors.KNeighborsClassifier(algorithm = 'ball_tree')
        knn4.fit(xtrain,ytrain)
        ytest = knn4.predict(xtest)
        predictionMat[:,count] = ytest
        predictionStringMat[1,count] = predVec2Str(ytest)
        count+=1


    if count < numfiers:
        knn5 = neighbors.KNeighborsClassifier(algorithm = 'kd_tree')
        knn5.fit(xtrain,ytrain)
        ytest = knn5.predict(xtest)
        predictionMat[:,count] = ytest
        predictionStringMat[1,count] = predVec2Str(ytest)
        count+=1



    if count < numfiers:
        ncc1 = NearestCentroid()
        ncc1.fit(xtrain,ytrain)
        ytest = ncc1.predict(xtest)
        predictionMat[:,count] = ytest
        predictionStringMat[1,count] = predVec2Str(ytest)
        count+=1

    if count < numfiers:
        tree5 = ABC()
        tree5.fit(xtrain,ytrain)
        ytest = tree5.predict(xtest)
        predictionMat[:,count] = ytest
        predictionStringMat[1,count] = predVec2Str(ytest)
        count+=1




    return (predictionMat,predictionStringMat)




def predVec2Str(ytest):
    gridLetters = 'ABCDEFGHI'
    str = ''
    for pred in ytest:
        #remember, A corresponds to class 1
        str += gridLetters[pred-1]
    return str
