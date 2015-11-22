
# coding: utf-8

# In[24]:

from sklearn import datasets
import math
import scipy
import numpy as np
from scipy.io import wavfile

def get_all_features(wlength=10000):
    recordingTypeLetters = 'AP'
    recordingTypeNames = ['Audio_recordings','Power_recordings']
    recording_file = ['Aud','Pow']
    numRecordingsPerGrid = [2,9,2,10,2,11,2,11,2,11,2,8,2,11,2,11,2,11]
    numPowerRecsPerGrid = numRecordingsPerGrid[1::2]
    numAudioRecsPerGrid = numRecordingsPerGrid[0::2]
    print numPowerRecsPerGrid


    featMatAudio = []
    featMatPower = []
    indVecPower = []
    indVecAudio = []

    trainingGridLetters = 'ABCDEFGHI'
    count = 0

    for recIndex,tempNumRecordings in enumerate(numRecordingsPerGrid):

        recType = recordingTypeLetters[recIndex%2]
        recTypeName = recordingTypeNames[recIndex%2]
        recording_file_name = recording_file[recIndex%2]
        trainingGrid = trainingGridLetters[int(math.floor(recIndex/2))]

        tempCell = [0 for i in range(tempNumRecordings)]

        for i in range(tempNumRecordings):
            tempStr = '../IEEEDataset/Grid_' + trainingGrid + '/' + recTypeName + '/Train_Grid_' + trainingGrid + '_' + recType + str(i+1) + '.wav'
            print tempStr

            fs, tempSig = wavfile.read(tempStr)
            count = count+ 1
            #All_recordings{count} = x;

            #tempCell{ii} = x;
            tempCell[i] = tempSig;

            feature_set = get_features(tempSig,wlength)


            if recIndex%2 == 0: #AUDIO
                indVecAudio.append(feature_set.shape[0])
                tempclassVec = (np.zeros((feature_set.shape[0],1))+recIndex/2)
                feature_set = np.append(tempclassVec,feature_set,axis=1)

                if len(featMatAudio) == 0:
                    featMatAudio = feature_set
                else:
                    featMatAudio = np.append(featMatAudio,feature_set,axis=0)
            else: #POWER
                indVecPower.append(feature_set.shape[0])
                tempclassVec = (np.zeros((feature_set.shape[0],1))+(recIndex-1)/2)
                feature_set = np.append(tempclassVec,feature_set,axis=1)
                if len(featMatPower) == 0:
                    featMatPower = feature_set
                else:
                   featMatPower = np.append(featMatPower,feature_set,axis=0)


    return (featMatPower,featMatAudio,indVecPower,numPowerRecsPerGrid,indVecAudio,numAudioRecsPerGrid)







# In[25]:

def get_features(signal,wlength=10000):
    windowedSig = np.reshape(signal,(signal.shape[0]/wlength,wlength))

    #TIME DOMAIN

    #Integrated ENF Sig
    IENF_Tr = np.sum(np.abs(windowedSig),1)
    #Mean Abs Val
    MAV_Tr = np.mean(np.abs(windowedSig),1)

    #Mean Abs Value Slope - gives the difference between MAVs of adjacent segments
    MAVS_Tr = np.diff(MAV_Tr)
    #Simple Square Integral - total power per window
    SSI_Tr = np.sum(np.power(np.abs(windowedSig),2),1)
    #Variance
    Var_Tr = np.var(windowedSig,1)
    #RMS - root mean square
    RMS_Tr = np.sqrt(np.abs(np.mean(np.power(windowedSig,2),1)))
    #Waveform Length - cumulative length of the waveform over the time segment
    WL_Tr = np.sum(np.abs(np.diff(windowedSig,axis=1)),1)

    #FREQUENCY DOMAIN

    #FFT
    FFT_Tr = np.fft.fft(windowedSig)
    #Mean Frequency

    #Mean Absolute Value - Freq
    MAVFreq_Tr = np.mean(np.abs(FFT_Tr),1)
    # Mean Absolute Value Slope - Frequency
    MAVSFreq_Tr = np.diff(MAVFreq_Tr)
    #Maximum frequency
    MaxFreq_Tr = np.max(np.abs(FFT_Tr),1)
    #Frequency Variance
    VarFreq_Tr = np.var(FFT_Tr,1)
    #Frequency RMS
    RMSFreq_Tr = np.abs(np.sqrt(np.mean(np.power(FFT_Tr,2),1)))

    feature_set = np.array([IENF_Tr[:-1],MAV_Tr[:-1],MAVS_Tr,SSI_Tr[:-1],Var_Tr[:-1],RMS_Tr[:-1],                             WL_Tr[:-1],MAVFreq_Tr[:-1],MAVSFreq_Tr,MaxFreq_Tr[:-1],VarFreq_Tr[:-1],RMSFreq_Tr[:-1]])

    feature_set = feature_set.T

    return feature_set



# In[73]:

#takes feature matrix of N features and M samples (featMat is MxN+1)
def splitFeatMat(featMat,indVec,numRecsPerGrid,numTestingRecsPerGrid):
    trainingMat = np.empty((0,featMat.shape[1]))
    testingMat = np.empty((0,featMat.shape[1]))

    recCount = 0
    for numRecs in numRecsPerGrid:
        startRecInd = recCount
        endRecInd = recCount + numRecs - 1

        tempRecInds = np.array([])
        for i in range(startRecInd,endRecInd+1):
            tempRecInds = np.append(tempRecInds,i)

        #np.random.shuffle(tempRecInds)

        testRecInds = tempRecInds[0:numTestingRecsPerGrid]
        trainRecInds = tempRecInds[numTestingRecsPerGrid:]
        
        featureRowCount = 0
        for tempTestRecInd in testRecInds:
            tempTestRecInd = int(tempTestRecInd)
            try:
                start = indVec[tempTestRecInd]+featureRowCount
                end = indVec[tempTestRecInd+1]+featureRowCount
                #print "Start " + str(start)
                #print "End " + str(end)
                tempMat = featMat[start:end,:]
                #print "TestMatShape " + str(testingMat.shape)
                #print "TempMatShape " + str(tempMat.shape)
                testingMat = np.append(testingMat,tempMat,0)
                featureRowCount += (indVec[tempTestRecInd+1]-indVec[tempTestRecInd])
            except IndexError:
                testingMat = np.append(testingMat,featMat[indVec[tempTestRecInd]+featureRowCount:,:],0)
                featureRowCount += (featMat.shape[0]-(indVec[tempTestRecInd]+featureRowCount))
           
        featureRowCount = 0
        for tempTrainRecInd in trainRecInds:
            tempTrainRecInd = int(tempTrainRecInd)
            try:
                trainingMat = np.append(trainingMat,featMat[indVec[tempTrainRecInd]+featureRowCount:indVec[tempTrainRecInd+1]+featureRowCount,:],0)
                featureRowCount += (indVec[tempTrainRecInd+1]-indVec[tempTrainRecInd])
            except IndexError:
                trainingMat = np.append(trainingMat,featMat[indVec[tempTrainRecInd]+featureRowCount:,:],0)
                featureRowCount += (featMat.shape[0]-(indVec[tempTrainRecInd]+featureRowCount))
        
        
            

        recCount += numRecs
        xtrain = trainingMat[:,1:]
        ytrain = trainingMat[:,0]
        xtest = testingMat[:,1:]
        ytest = testingMat[:,0]
    return (xtrain,ytrain,xtest,ytest,trainingMat,testingMat)


# In[74]:

import numpy as np
import scipy.io as sio


ytrain = np.ravel(ytrain)
ytest = np.ravel(ytest)


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
def myclassify(numfiers=5,xtrain=xtrain,ytrain=ytrain,xtest=xtest,ytest=ytest):
    count = 0



    bagging2 = BaggingClassifier(ETC(),bootstrap=False,bootstrap_features=False)
    bagging2.fit(xtrain,ytrain)
    #print bagging2.score(xtest,ytest)
    count += 1
    classifiers = [bagging2.score(xtest,ytest)]

    if count < numfiers:

        tree2 = ETC()
        tree2.fit(xtrain,ytrain)
        #print tree2.fit(xtrain,ytrain)
        #print tree2.score(xtest,ytest)
        count+=1
        classifiers = np.append(classifiers,tree2.score(xtest,ytest))
        print "1"
        print tree2.score(xtest,ytest)

    if count < numfiers:
        bagging1 = BaggingClassifier(ETC())
        bagging1.fit(xtrain,ytrain)
        #print bagging1.score(xtest,ytest)
        count+=1
        classifiers = np.append(classifiers,bagging1.score(xtest,ytest))
        print "2"
        print bagging1.score(xtest,ytest)

#     if count < numfiers:
#         # votingClassifiers combine completely different machine learning classifiers and use a majority vote
#         clff1 = SVC()
#         clff2 = RFC(bootstrap=False)
#         clff3 = ETC()
#         clff4 = neighbors.KNeighborsClassifier()
#         clff5 = quadda()
#         print"3"


#         eclf = VotingClassifier(estimators = [('svc',clff1),('rfc',clff2),('etc',clff3),('knn',clff4),('qda',clff5)])
#         eclf = eclf.fit(xtrain,ytrain)
#         #print(eclf.score(xtest,ytest))
#         # for claf, label in zip([clff1,clff2,clff3,clff4,clff5,eclf],['SVC','RFC','ETC','KNN','QDA','Ensemble']):
#         #     cla
#         #     scores = crossvalidation.cross_val_score(claf,xtrain,ytrain,scoring='accuracy')
#         #     print ()
#         count+=1
#         classifiers = np.append(classifiers,eclf.score(xtest,ytest))


#     if count < numfiers:
#         svc1 = SVC()
#         svc1.fit(xtrain,ytrain)
#         dec = svc1.score(xtest,ytest)
#         count+=1
#         classifiers = np.append(classifiers,svc1.score(xtest,ytest))
#         print "3"

    if count < numfiers:
        # Quadradic discriminant analysis - classifier with quadratic decision boundary -
        qda = quadda()
        qda.fit(xtrain,ytrain)
        #print(qda.score(xtest,ytest))
        count+=1
        classifiers = np.append(classifiers,qda.score(xtest,ytest))
        print "4"


    if count < numfiers:

        tree1 = DTC()
        tree1.fit(xtrain,ytrain)
        #print tree1.fit(xtrain,ytrain)
        #print tree1.score(xtest,ytest)
        count+=1
        classifiers = np.append(classifiers,tree1.score(xtest,ytest))

    if count < numfiers:
        knn1 = neighbors.KNeighborsClassifier() # this classifies based on the #k nearest neighbors, where k is definted by the user.
        knn1.fit(xtrain,ytrain)
        #print(knn1.score(xtest,ytest))
        count+=1
        classifiers = np.append(classifiers,knn1.score(xtest,ytest))

    if count < numfiers:
        # linear discriminant analysis - classifier with linear decision boundary -
        lda = linda()
        lda.fit(xtrain,ytrain)
        #print(lda.score(xtest,ytest))
        count+=1
        classifiers = np.append(classifiers,lda.score(xtest,ytest))

    if count < numfiers:
        tree3 = RFC()
        tree3.fit(xtrain,ytrain)
        #print tree3.score(xtest,ytest)
        count+=1
        classifiers = np.append(classifiers,tree3.score(xtest,ytest))

    if count < numfiers:
        bagging3 = BaggingClassifier(RFC(),bootstrap=False,bootstrap_features=False)
        bagging3.fit(xtrain,ytrain)
        #print bagging3.score(xtest,ytest)
        count+=1
        classifiers = np.append(classifiers,bagging3.score(xtest,ytest))


    if count < numfiers:
        bagging4 = BaggingClassifier(SVC(),bootstrap=False,bootstrap_features=False)
        bagging4.fit(xtrain,ytrain)
        #print bagging4.score(xtest,ytest)
        count+=1
        classifiers = np.append(classifiers,bagging4.score(xtest,ytest))

    if count < numfiers:
        tree4 = RFC(bootstrap=False)
        tree4.fit(xtrain,ytrain)
        #print tree4.score(xtest,ytest)
        count+=1
        classifiers = np.append(classifiers,tree4.score(xtest,ytest))

    if count < numfiers:
        tree6 = GBC()
        tree6.fit(xtrain,ytrain)
        #print(tree6.score(xtest,ytest))
        count+=1
        classifiers = np.append(classifiers,tree6.score(xtest,ytest))

    if count < numfiers:
        knn2 = neighbors.KNeighborsClassifier(n_neighbors = 10)
        knn2.fit(xtrain,ytrain)
        #print(knn2.score(xtest,ytest))
        count+=1
        classifiers = np.append(classifiers,knn2.score(xtest,ytest))

    if count < numfiers:
        knn3 = neighbors.KNeighborsClassifier(n_neighbors = 3)
        knn3.fit(xtrain,ytrain)
        #print(knn3.score(xtest,ytest))
        count+=1
        classifiers = np.append(classifiers,knn3.score(xtest,ytest))

    if count < numfiers:
        knn4 = neighbors.KNeighborsClassifier(algorithm = 'ball_tree')
        knn4.fit(xtrain,ytrain)
        #print(knn4.score(xtest,ytest))
        count+=1
        classifiers = np.append(classifiers,knn4.score(xtest,ytest))

    if count < numfiers:
        knn5 = neighbors.KNeighborsClassifier(algorithm = 'kd_tree')
        knn5.fit(xtrain,ytrain)
        #print(knn5.score(xtest,ytest))
        count+=1
        classifiers = np.append(classifiers,knn5.score(xtest,ytest))

    if count < numfiers:
        ncc1 = NearestCentroid()
        ncc1.fit(xtrain,ytrain)
        #print (ncc1.score(xtest,ytest))
        count+=1
        classifiers = np.append(classifiers,ncc1.score(xtest,ytest))

    if count < numfiers:
    # Nearest shrunken Centroid
        for shrinkage in [None,0.05,0.1,0.2,0.3,0.4,0.5]:
            ncc2 = NearestCentroid(shrink_threshold = shrinkage)
            ncc2.fit(xtrain,ytrain)
            #print(ncc2.score(xtest,ytest))

        count+=1
        classifiers = np.append(classifiers,ncc2.score(xtest,ytest))

    if count < numfiers:
        tree5 = ABC()
        tree5.fit(xtrain,ytrain)
        #print(tree5.score(xtest,ytest))
        count+=1
        classifiers = np.append(classifiers,tree5.score(xtest,ytest))

    classifierlabel = ["BaggingETC (with bootstraps set to false)","ETC","BaggingETC","Voting Classifier","svm","QDA","DTC","KNN (default)","LDA","RFC",
                       "BaggingRFC (with bootstraps set to false)","BaggingSVC (with bootstraps set to false)","RFC (bootstrap false)","GBC",
                        "knn (n_neighbors = 10)","knn (n_neighbors = 3)","knn (ball tree algorithm)","knn (kd_tree algorithm)",
                       "Nearest Centroid","Shrunken Centroid?","ABC"]


    classifierlabel = classifierlabel[:len(classifiers)]
    #print len(classifiers)
    #print classifiers
    for i in range(len(classifiers)):


        print ("{} classifier has percent correct {}".format(classifierlabel[i],classifiers[i]))


# In[75]:

featMatPower = np.load("featMatPower.npy")
indVecPower = np.load("indVecPower.npy")
numPowerRecsPerGrid = np.load("numPowerRecsPerGrid.npy")
print indVecPower
print numPowerRecsPerGrid


# In[91]:

#example of how to run it
#featMatPower,featMatAudio,indVecPower,numPowerRecsPerGrid,indVecAudio,numAudioRecsPerGrid= get_all_features()
xtrain,ytrain,xtest,ytest,trainingMat,testingMat = splitFeatMat(featMatPower,indVecPower,numPowerRecsPerGrid,6)
print trainingMat.shape
print xtrain.shape
print testingMat.shape
print xtest.shape
print ytrain[1:15]


# In[92]:

myclassify(5,xtrain=xtrain,ytrain=ytrain,xtest=xtest,ytest=ytest)


# In[ ]:



