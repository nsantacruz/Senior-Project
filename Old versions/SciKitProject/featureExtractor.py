from sklearn import datasets
import math
import scipy
import numpy as np
from scipy.io import wavfile

def get_all_features(wlength=1000):
    recordingTypeLetters = 'AP'
    recordingTypeNames = ['Audio_recordings','Power_recordings']
    recording_file = ['Aud','Pow']
    numRecordingsPerGrid = [2,9,2,10,2,11,2,11,2,11,2,8,2,11,2,11,2,11]
    numPowerRecsPerGrid = numRecordingsPerGrid[1::2]
    numAudioRecsPerGrid = numRecordingsPerGrid[0::2]
    print numPowerRecsPerGrid


    featMatAudio = np.empty((0,13))
    featMatPower = np.empty((0,13))
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
                featMatAudio = np.append(featMatAudio,feature_set,axis=0)

            else: #POWER
                indVecPower.append(feature_set.shape[0])
                tempclassVec = (np.zeros((feature_set.shape[0],1))+(recIndex-1)/2)
                feature_set = np.append(tempclassVec,feature_set,axis=1)
                featMatPower = np.append(featMatPower,feature_set,axis=0)


    return (featMatPower,featMatAudio,indVecPower,numPowerRecsPerGrid,indVecAudio,numAudioRecsPerGrid)




def get_features(signal,wlength=1000):
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

    feature_set = np.array([IENF_Tr[:-1],MAV_Tr[:-1],MAVS_Tr,SSI_Tr[:-1],Var_Tr[:-1],RMS_Tr[:-1], \
                            WL_Tr[:-1],MAVFreq_Tr[:-1],MAVSFreq_Tr,MaxFreq_Tr[:-1],VarFreq_Tr[:-1],RMSFreq_Tr[:-1]])

    feature_set = feature_set.T

    return feature_set

#takes feature matrix of N features and M samples (featMat is MxN+1)
def splitFeatMat(featMat,indVec,numRecsPerGrid,numTestingRecsPerGrid):
    trainingMat = np.array([])
    testingMat = np.array([])

    recCount = 0
    featureRowCount = 0
    for numRecs in numRecsPerGrid:
        startRecInd = recCount
        endRecInd = recCount + numRecs - 1

        tempRecInds = np.array([])
        for i in range(startRecInd,endRecInd+1):
            tempRecInds = np.append(tempRecInds,i)

        np.random.shuffle(tempRecInds)

        testRecInds = tempRecInds[0:numTestingRecsPerGrid]
        trainRecInds = tempRecInds[numTestingRecsPerGrid:]

        for tempTestRecInd in testRecInds:
            tempTestRecInd = int(tempTestRecInd)
            try:
                testingMat = np.append(testingMat,featMat[indVec[tempTestRecInd]+featureRowCount:indVec[tempTestRecInd+1]+featureRowCount,:])
                featureRowCount += (indVec[tempTestRecInd+1]-indVec[tempTestRecInd])
            except IndexError:
                testingMat = np.append(testingMat,featMat[indVec[tempTestRecInd]+featureRowCount:,:])



        for tempTrainRecInd in trainRecInds:
            tempTrainRecInd = int(tempTrainRecInd)
            try:
                trainingMat = np.append(trainingMat,featMat[indVec[tempTrainRecInd]+featureRowCount:indVec[tempTrainRecInd+1]+featureRowCount,:])
                featureRowCount += (indVec[tempTrainRecInd+1]-indVec[tempTrainRecInd])
            except IndexError:
                trainingMat = np.append(trainingMat,featMat[indVec[tempTrainRecInd]+featureRowCount:,:])




        recCount += numRecs

    return (trainingMat,testingMat)









#example of how to run it
featMatPower,featMatAudio,indVecPower,numPowerRecsPerGrid,indVecAudio,numAudioRecsPerGrid= get_all_features()

np.save("featMatPower.npy",featMatPower)
np.save("indVecPower.npy",indVecPower)
np.save("numPowerRecsPerGrid.npy",numPowerRecsPerGrid)
#trainingMat,testingMat = splitFeatMat(featMatPower,indVecPower,numPowerRecsPerGrid,2)

#yo = 4



