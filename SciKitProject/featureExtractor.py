from sklearn import datasets
import math
import scipy
import numpy as np
from scipy.io import wavfile

def get_features(wlength=10000):
    recordingTypeLetters = 'AP'
    recordingTypeNames = ['Audio_recordings','Power_recordings']
    recording_file = ['Aud','Pow']
    numRecordingsPerGrid = [2,9,2,10,2,11,2,11,2,11,2,8,2,11,2,11,2,11]
    numRecordingsPerGrid = [0,9]



    featMatAudio = []
    featMatPower = []

    trainingGridLetters = 'ABCDEFGHI'
    count = 0
    #All_recordings = cell(sum(numRecordingsPerGrid),1)

    for recIndex,tempNumRecordings in enumerate(numRecordingsPerGrid):

        recType = recordingTypeLetters[recIndex%2]
        recTypeName = recordingTypeNames[recIndex%2]
        recording_file_name = recording_file[recIndex%2]
        trainingGrid = trainingGridLetters[int(math.floor(recIndex/2))]

        #tempCell = cell(endofii,1)
        tempCell = [0 for i in range(tempNumRecordings)]

        for i in range(tempNumRecordings):
            tempStr = '../IEEEDataset/Grid_' + trainingGrid + '/' + recTypeName + '/Train_Grid_' + trainingGrid + '_' + recType + str(i+1) + '.wav'
            print tempStr

            fs, tempSig = wavfile.read(tempStr)
            count = count+ 1
            #All_recordings{count} = x;

            #tempCell{ii} = x;
            tempCell[i] = tempSig;

            windowedSig = np.reshape(tempSig,(tempSig.shape[0]/wlength,wlength))

            '''
                        %Integrated ENF Signal
            IENF_Tr = sum(abs(tempTest_Tr));
            %Mean Absolute Value
            MAV_Tr = mean(abs(tempTest_Tr));
            % Mean Absolute Value Slope - gives the difference beween MAVs of adjacent
            % segments
            MAVS_Tr = diff(MAV_Tr);
            %Simple Square Integral - total power per window
            SSI_Tr = sum(abs(tempTest_Tr).^2);
            %Variance
            Var_Tr = var(tempTest_Tr,0);%,%,2); %0 makes this unweighted, so it is equal to var(x)
            %RMS - root mean square
            RMS_Tr = sqrt(mean(tempTest_Tr.^2));
            % Waveform Length - cumulative length of the waveform over the time segment
            WL_Tr = sum(abs(diff(tempTest_Tr)));
            '''
            #start extracting features

            #Integrated ENF Sig
            IENF_Tr = np.sum(np.abs(windowedSig))
            #Mean Abs Val
            MAV_Tr = np.mean(np.abs(windowedSig),1)

            #Mean Abs Value Slope - gives the difference between MAVs of adjacent segments
            MAVS_Tr = np.diff(MAV_Tr)
            #Simple Square Integral - total power per window
            SSI_Tr = np.sum(np.power(np.abs(windowedSig),2))
            #Variance
            Var_Tr = np.var(windowedSig)
            #RMS - root mean square
            RMS_Tr = np.sqrt(np.mean(np.power(windowedSig,2)))
            #Waveform Length - cumulative length of the waveform over the time segment
            WL_Tr = np.sum(np.abs(np.diff(windowedSig)))






    #np.save(trainingGrid + '_' + recording_file_name + '.npy',tempCell)






get_features()



