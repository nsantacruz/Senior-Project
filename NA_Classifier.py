


import numpy as np
from sklearn.svm import OneClassSVM as oneclass
import scipy.io as sio
import Transfer_Mat_From_Matlab
from Transfer_Mat_From_Matlab import txmat


from sklearn.covariance import EllipticEnvelope as EllipticEnv



def myclassify_NA(numfiers,xtrain,xtest,xtrunc,xtest2,xtrunc2,nuparam):

    # remove NaN, Inf, and -Inf values from the xtest feature matrix
    xtest = xtest[~np.isnan(xtest).any(axis=1),:]
    xtest = xtest[~np.isinf(xtest).any(axis=1),:]

    xtrunclength = sio.loadmat('xtrunclength.mat')
    xtrunclength = xtrunclength['xtrunclength'][0]
    xtrunc = xtrunc[0]
    xtrunc2 = xtrunc2[0]
    #if xtest is NxM matrix, returns Nxnumifiers matrix where each column corresponds to a classifiers prediction vector
    count = 0
    print numfiers

    predictionMat = np.empty((xtest.shape[0],numfiers))
    predictionStringMat = []
    predictionMat2 = np.empty((xtest2.shape[0],numfiers))
    predictionStringMat2 = []

    print 'part 1'

    oneclassclass = oneclass(nu = nuparam)
    print 'part 2'
    oneclassclass.fit(xtrain)
    print 'part 3'
    ytest = oneclassclass.predict(xtest)
    ytest2 = oneclassclass.predict(xtest2)
    print 'part 4'
    predictionMat[:,count] = ytest
    predictionMat2[:,count] = ytest2
    count += 1
    #print oneclassclass.get_params()

    # print 'finished one'
    # if count < numfiers:
    #     oc2 = oneclass(nu = .05)
    #     oc2.fit(xtrain)
    #     ytest = oc2.predict(xtest)
    #     predictionMat[:,count] = ytest
    #     count +=1
    # #
    # if count < numfiers:
    #     oc3 = oneclass(kernel='linear', degree = 4)
    #     oc3.fit(xtrain)
    #     ytest = oc3.predict(xtest)
    #     predictionMat[:,count] = ytest
    #     count+=1
    #
    # if count < numfiers:
    #     oneclass2 = oneclass(kernel = 'poly', degree = 4)
    #     print 'part 5'
    #     oneclass2.fit(xtrain)
    #     print 'part 6'
    #     ytest = oneclass2.predict(xtest)
    #     print 'part 7'
    #     predictionMat[:,count] = ytest
    #     count+=1
    if count < numfiers:
        oneclass2 = oneclass(kernel = 'poly', nu = nuparam)
        print 'part 5'
        oneclass2.fit(xtrain)
        print 'part 6'
        ytest = oneclass2.predict(xtest)
        ytest2 = oneclass2.predict(xtest2)
        print 'part 7'
        predictionMat[:,count] = ytest
        predictionMat2[:,count] = ytest2
        count+=1

    for colCount in range(predictionMat.shape[1]):
        tempCol = predictionMat[:,colCount]
        modeCol = predWindowVecModeFinder(tempCol,xtrunc)
        modeStr = predVec2Str(modeCol)
        predictionStringMat.append(modeStr)

    for colCount2 in range(predictionMat2.shape[1]):
        tempCol2 = predictionMat2[:,colCount2]
        modeCol2 = predWindowVecModeFinder(tempCol2,xtrunc2)
        modeStr2 = predVec2Str(modeCol2)
        predictionStringMat2.append(modeStr2)


    return predictionStringMat,predictionStringMat2


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

    #OneClassSVM.predict(xtest) returns 1 if the classifier believes
    # that the test sample is from the training dataset, and -1 if not
    stri = ''
    for pred in ytest:
        stri += (' ' + str(int(pred)))
        #remember, 1 corresponds to yes, -1 to no
        # new = 1 if int(pred)== 1 else 0
        # str+=gridLetters[new]
        #str += gridLetters[int(pred)-1]
    return stri











    # oneclassclass = oneclass()
    # oneclassclass.fit(xtrain)
    # ytest = oneclassclass.predict(xtest)
    # predictionMat[:,count] = ytest
    # count += 1
    # #print oneclassclass.get_params()
    #
    #
    # if count < numfiers:
    #     oc2 = oneclass(degree=4)
    #     oc2.fit(xtrain)
    #     ytest = oc2.predict(xtest)
    #     predictionMat[:,count] = ytest
    #     count +=1
    #
    #
    # if count < numfiers:
    #     oneclass2 = oneclass(degree=5)
    #     oneclass2.fit(xtrain)
    #     ytest = oneclass2.predict(xtest)
    #     predictionMat[:,count] = ytest
    #     count+=1
    #
    # if count < numfiers:
    #     oc3 = oneclass(kernel='linear')
    #     oc3.fit(xtrain)
    #     ytest = oc3.predict(xtest)
    #     predictionMat[:,count] = ytest
    #     count+=1
    #
    # if count < numfiers:
    #     oneclass2 = oneclass(kernel = 'poly')
    #     oneclass2.fit(xtrain)
    #     ytest = oneclass2.predict(xtest)
    #     predictionMat[:,count] = ytest
    #     count+=1
    #
    # if count < numfiers:
    #     oc3 = oneclass(kernel='sigmoid')
    #     oc3.fit(xtrain)
    #     ytest = oc3.predict(xtest)
    #     predictionMat[:,count] = ytest
    #     count+=1
    #
    # if count < numfiers:
    #     oc3 = oneclass(kernel='linear', degree = 4)
    #     oc3.fit(xtrain)
    #     ytest = oc3.predict(xtest)
    #     predictionMat[:,count] = ytest
    #     count+=1
    #
    # if count < numfiers:
    #     oneclass2 = oneclass(kernel = 'poly', degree = 4)
    #     oneclass2.fit(xtrain)
    #     ytest = oneclass2.predict(xtest)
    #     predictionMat[:,count] = ytest
    #     count+=1
    #
    # if count < numfiers:
    #     oc3 = oneclass(kernel='sigmoid', degree = 4)
    #     oc3.fit(xtrain)
    #     ytest = oc3.predict(xtest)
    #     predictionMat[:,count] = ytest
    #     count+=1


    # if count < numfiers:
    #     oneclass3 = oneclass(shrinking=1)
    #     oneclass3.fit(xtrain)
    #     ytest = oneclass3.predict(xtest)
    #     predictionMat[:,count] = ytest
    #     count+=1
    #
    # if count < numfiers:
    #     oneclass4 = oneclass(degree=2)
    #     oneclass4.fit(xtrain)
    #     ytest = oneclass4.predict(xtest)
    #     predictionMat[:,count] = ytest
    #     count+=1
    #




    # if count < numfiers:
    #     El_Env = EllipticEnv()
    #     El_Env.fit(xtrain)
    #     ytest =El_Env.predict(xtest)
    #     predictionMat[:,count] = ytest
    #     count+=1


