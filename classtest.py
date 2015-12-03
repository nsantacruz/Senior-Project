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


# This is a classification function to test our classifiers and feature sets
# numfiers represents the number of classifiers you would like to train. it takes an integer argument and will train
# up to ~20 classifiers. The larger the number of classifiers the longer the function will take to run
# xtrain represents your training feature set matrix (your matrix of predictor values). it takes an mxn matrix, where m = number of samples and n = number of features
# ytrain represents your target vector for xtrain. it takes an mx1 matrix, where m = number of samples
# xtest and ytest are the same as xtrain and ytrain but for a section of holdout data. the scores of the classifiers will be evaluated based
# on the performance of the classifiers against this set.

def myclassify(numfiers,xtrain,ytrain,xtest,ytest):
    count = 0
    print numfiers

    ytrain = np.ravel(ytrain)
    ytest = np.ravel(ytest)


    bagging2 = BaggingClassifier(ETC(),bootstrap=False,bootstrap_features=False)
    bagging2.fit(xtrain,ytrain)
    #print bagging2.score(xtest,ytest)
    count += 1
    classifiers = [bagging2.score(xtest,ytest)]
    print "percentage classifcation complete: %s" % str(round(100*(float(count)/numfiers))) + "%"


    if count < numfiers:

        tree2 = ETC()
        tree2.fit(xtrain,ytrain)
        #print tree2.fit(xtrain,ytrain)
        #print tree2.score(xtest,ytest)
        count+=1
        classifiers = np.append(classifiers,tree2.score(xtest,ytest))
        print "percentage classifcation complete: %s" % str(round(100*(float(count)/numfiers))) + "%" + "   " + str(numfiers-count) + "classifiers left to train"

    if count < numfiers:
        bagging1 = BaggingClassifier(ETC())
        bagging1.fit(xtrain,ytrain)
        #print bagging1.score(xtest,ytest)
        count+=1
        classifiers = np.append(classifiers,bagging1.score(xtest,ytest))
        print "percentage classifcation complete: %s" % str(round(100*(float(count)/numfiers))) + "%" + "   " + str(numfiers-count) + "classifiers left to train"

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
        count+=1
        classifiers = np.append(classifiers,eclf.score(xtest,ytest))
        print "percentage classifcation complete: %s" % str(round(100*(float(count)/numfiers))) + "%" + "   " + str(numfiers-count) + "classifiers left to train"


    if count < numfiers:
        svc1 = SVC()
        svc1.fit(xtrain,ytrain)
        dec = svc1.score(xtest,ytest)
        count+=1
        classifiers = np.append(classifiers,svc1.score(xtest,ytest))
        print "percentage classifcation complete: %s" % str(round(100*(float(count)/numfiers))) + "%" + "   " + str(numfiers-count) + "classifiers left to train"

    if count < numfiers:
        # Quadradic discriminant analysis - classifier with quadratic decision boundary -
        qda = quadda()
        qda.fit(xtrain,ytrain)
        #print(qda.score(xtest,ytest))
        count+=1
        classifiers = np.append(classifiers,qda.score(xtest,ytest))
        print "percentage classifcation complete: %s" % str(round(100*(float(count)/numfiers))) + "%" + "   " + str(numfiers-count) + "classifiers left to train"



    if count < numfiers:

        tree1 = DTC()
        tree1.fit(xtrain,ytrain)
        #print tree1.fit(xtrain,ytrain)
        #print tree1.score(xtest,ytest)
        count+=1
        classifiers = np.append(classifiers,tree1.score(xtest,ytest))
        print "percentage classifcation complete: %s" % str(round(100*(float(count)/numfiers))) + "%" + "   " + str(numfiers-count) + "classifiers left to train"

    if count < numfiers:
        knn1 = neighbors.KNeighborsClassifier() # this classifies based on the #k nearest neighbors, where k is definted by the user.
        knn1.fit(xtrain,ytrain)
        #print(knn1.score(xtest,ytest))
        count+=1
        classifiers = np.append(classifiers,knn1.score(xtest,ytest))
        print "percentage classifcation complete: %s" % str(round(100*(float(count)/numfiers))) + "%" + "   " + str(numfiers-count) + "classifiers left to train"


    if count < numfiers:
        # linear discriminant analysis - classifier with linear decision boundary -
        lda = linda()
        lda.fit(xtrain,ytrain)
        #print(lda.score(xtest,ytest))
        count+=1
        classifiers = np.append(classifiers,lda.score(xtest,ytest))
        print "percentage classifcation complete: %s" % str(round(100*(float(count)/numfiers))) + "%" + "   " + str(numfiers-count) + "classifiers left to train"

    if count < numfiers:
        tree3 = RFC()
        tree3.fit(xtrain,ytrain)
        #print tree3.score(xtest,ytest)
        count+=1
        classifiers = np.append(classifiers,tree3.score(xtest,ytest))
        print "percentage classifcation complete: %s" % str(round(100*(float(count)/numfiers))) + "%" + "   " + str(numfiers-count) + "classifiers left to train"

    if count < numfiers:
        bagging3 = BaggingClassifier(RFC(),bootstrap=False,bootstrap_features=False)
        bagging3.fit(xtrain,ytrain)
        #print bagging3.score(xtest,ytest)
        count+=1
        classifiers = np.append(classifiers,bagging3.score(xtest,ytest))
        print "percentage classifcation complete: %s" % str(round(100*(float(count)/numfiers))) + "%" + "   " + str(numfiers-count) + "classifiers left to train"


    if count < numfiers:
        bagging4 = BaggingClassifier(SVC(),bootstrap=False,bootstrap_features=False)
        bagging4.fit(xtrain,ytrain)
        #print bagging4.score(xtest,ytest)
        count+=1
        classifiers = np.append(classifiers,bagging4.score(xtest,ytest))
        print "percentage classifcation complete: %s" % str(round(100*(float(count)/numfiers))) + "%" + "   " + str(numfiers-count) + "classifiers left to train"

    if count < numfiers:
        tree4 = RFC(bootstrap=False)
        tree4.fit(xtrain,ytrain)
        #print tree4.score(xtest,ytest)
        count+=1
        classifiers = np.append(classifiers,tree4.score(xtest,ytest))
        print "percentage classifcation complete: %s" % str(round(100*(float(count)/numfiers))) + "%" + "   " + str(numfiers-count) + "classifiers left to train"

    if count < numfiers:
        tree6 = GBC()
        tree6.fit(xtrain,ytrain)
        #print(tree6.score(xtest,ytest))
        count+=1
        classifiers = np.append(classifiers,tree6.score(xtest,ytest))
        print "percentage classifcation complete: %s" % str(round(100*(float(count)/numfiers))) + "%" + "   " + str(numfiers-count) + "classifiers left to train"

    if count < numfiers:
        knn2 = neighbors.KNeighborsClassifier(n_neighbors = 10)
        knn2.fit(xtrain,ytrain)
        #print(knn2.score(xtest,ytest))
        count+=1
        classifiers = np.append(classifiers,knn2.score(xtest,ytest))
        print "percentage classifcation complete: %s" % str(round(100*(float(count)/numfiers))) + "%" + "   " + str(numfiers-count) + "classifiers left to train"

    if count < numfiers:
        knn3 = neighbors.KNeighborsClassifier(n_neighbors = 3)
        knn3.fit(xtrain,ytrain)
        #print(knn3.score(xtest,ytest))
        count+=1
        classifiers = np.append(classifiers,knn3.score(xtest,ytest))
        print "percentage classifcation complete: %s" % str(round(100*(float(count)/numfiers))) + "%" + "   " + str(numfiers-count) + "classifiers left to train"

    if count < numfiers:
        knn4 = neighbors.KNeighborsClassifier(algorithm = 'ball_tree')
        knn4.fit(xtrain,ytrain)
        #print(knn4.score(xtest,ytest))
        count+=1
        classifiers = np.append(classifiers,knn4.score(xtest,ytest))
        print "percentage classifcation complete: %s" % str(round(100*(float(count)/numfiers))) + "%" + "   " + str(numfiers-count) + "classifiers left to train"

    if count < numfiers:
        knn5 = neighbors.KNeighborsClassifier(algorithm = 'kd_tree')
        knn5.fit(xtrain,ytrain)
        #print(knn5.score(xtest,ytest))
        count+=1
        classifiers = np.append(classifiers,knn5.score(xtest,ytest))
        print "percentage classifcation complete: %s" % str(round(100*(float(count)/numfiers))) + "%" + "   " + str(numfiers-count) + "classifiers left to train"

    if count < numfiers:
        ncc1 = NearestCentroid()
        ncc1.fit(xtrain,ytrain)
        #print (ncc1.score(xtest,ytest))
        count+=1
        classifiers = np.append(classifiers,ncc1.score(xtest,ytest))
        print "percentage classifcation complete: %s" % str(round(100*(float(count)/numfiers))) + "%" + "   " + str(numfiers-count) + "classifiers left to train"

    if count < numfiers:
    # Nearest shrunken Centroid
        for shrinkage in [None,0.05,0.1,0.2,0.3,0.4,0.5]:
            ncc2 = NearestCentroid(shrink_threshold = shrinkage)
            ncc2.fit(xtrain,ytrain)
            #print(ncc2.score(xtest,ytest))

        count+=1
        classifiers = np.append(classifiers,ncc2.score(xtest,ytest))
        print "percentage classifcation complete: %s" % str(round(100*(float(count)/numfiers))) + "%" + "   " + str(numfiers-count) + "classifiers left to train"

    if count < numfiers:
        tree5 = ABC()
        tree5.fit(xtrain,ytrain)
        #print(tree5.score(xtest,ytest))
        count+=1
        classifiers = np.append(classifiers,tree5.score(xtest,ytest))
        print "percentage classifcation complete: %s" % str(round(100*(float(count)/numfiers))) + "%"

    classifierlabel = ["BaggingETC (with bootstraps set to false)","ETC","BaggingETC","Voting Classifier","svm","QDA","DTC","KNN (default)","LDA","RFC",
                       "BaggingRFC (with bootstraps set to false)","BaggingSVC (with bootstraps set to false)","RFC (bootstrap false)","GBC",
                        "knn (n_neighbors = 10)","knn (n_neighbors = 3)","knn (ball tree algorithm)","knn (kd_tree algorithm)",
                       "Nearest Centroid","Shrunken Centroid?","ABC"]


    classifierlabel = classifierlabel[:len(classifiers)]

    for i in range(len(classifiers)):


        print ("{} classifier has percent correct {}".format(classifierlabel[i],classifiers[i]))








#
# print "Hi"
#
# myclassify(6,xtrain,ytrain1,xtest,ytest1)
# myclassify(12,xtrain,ytrain1,xtest,ytest1)
