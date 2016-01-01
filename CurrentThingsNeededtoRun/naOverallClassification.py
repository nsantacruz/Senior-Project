import numpy as np

import FinalClassifier
import NA_Classifier
from CurrentThingsNeededtoRun import AudiovsPower
from CurrentThingsNeededtoRun.Transfer_Mat_From_Matlab import txmat



def lookatme():


    xtesting = txmat('xtesting.mat','xtesting')
    xtrunclength = txmat('xtrunclength.mat','xtrunclength')

    xtrainallpow = txmat('xtrain_all_pow.mat','xtrain')
    xtltrainallpow = txmat('xtltrain_all_pow.mat','xtltrain')
    ytrainallpow = txmat('ytrain_all_pow.mat','ytrain')


    # print "power results"
    y,x,power9Class = FinalClassifier.myclassify_practice_set(1, xtrainallpow, ytrainallpow, xtltrainallpow, xtrunclength, xtesting)
    # print y
    # print power9Class
    # print ""
    # print FinalClassifier.predVec2Str(power9Class)
    # print ""

    xtrainallaud = txmat('xtrain_all_aud.mat','xtrain')
    xtltrainallaud = txmat('xtltrain_all_aud.mat','xtltrain')
    ytrainallaud = txmat('ytrain_all_aud.mat','ytrain')

    # print 'audio results'
    # print ""

    y,x,audio9Class = FinalClassifier.myclassify_practice_set(1, xtrain = xtrainallaud, ytrain = ytrainallaud, xtltrain = xtltrainallaud, xtltest = xtrunclength, xtest = xtesting)
    # print y
    # print FinalClassifier.predVec2Str(audio9Class)
    # print ""


    ybintestallaud = txmat('ybintrain_all_aud.mat','ybintrain')
    ybintestallpow = txmat('ybintrain_all_pow.mat','ybintrain')


    y,audVpow = AudiovsPower.myclassify_AudPow(1, xtrainallaud, xtrainallpow, ybintestallaud, ybintestallpow, xtesting)
    # print 'results from binary audio power classifier '
    # print ""
    # print y
    # print ""
    # print audVpow



    grids = ['A','B','C','D','E','F','G','H','I']
    pow1ClassMat = np.empty([len(grids),len(audVpow)])


    #POWER ONE-CLASS
    xtrain = []
    xtltrain = []



    for grid in grids:
        xtrain.append(txmat('xtrain_' + grid + '_pow.mat','xtrain'))
        xtltrain.append(txmat('xtltrain_' + grid + '_pow.mat','xtltrain'))


    for i in range(len(xtrain)):
        ystring,yvec = NA_Classifier.myclassify_oneclass(1, xtrain[i], xtesting, xtrunclength, nuparam = .05)
        pow1ClassMat[i,:] = yvec
        # print 'results on training set for training on ' + grids[i] + ' power '
        # print ystring
    # print pow1ClassMat


    #AUDIO ONE-CLASS
    # grids = ['A_18class1','B_18class1','C_18class1','D_18class1','E_18class1','F_18class1','G_18class1','H_18class1','I_18class1',
    #          'A_18class2','B_18class2','C_18class2','D_18class2','E_18class2','F_18class2','G_18class2','H_18class2','I_18class2']

    aud1ClassMat = np.empty([len(grids),len(audVpow)])
    xtrain = []
    xtltrain = []
    for grid in grids:
        xtrain.append(txmat('xtrain_' + grid + '_aud.mat','xtrain'))
        xtltrain.append(txmat('xtltrain_' + grid + '_aud.mat','xtltrain'))


    for i in range(len(xtrain)):
        ystring,yvec = NA_Classifier.myclassify_oneclass(1, xtrain[i], xtesting, xtrunclength, nuparam = .1)
        aud1ClassMat[i,:] = yvec
        # print 'results on training set for training on ' + grids[i] + ' audio '
        # print ystring
    # print aud1ClassMat.shape


    # we should run it for audio through 18 one class classifiers, one off each recording


    xtrain_BD_aud = txmat('xtrain_BD_aud.mat','xtrain')
    ytrain_BD_aud = txmat('ytrain_BD_aud.mat','ytrain')
    xtltrain_BD_aud = txmat('xtltrain_BD_aud.mat','xtltrain')
    y,x,BDsvm = FinalClassifier.myclassify_practice_set(1,xtrain_BD_aud,ytrain_BD_aud,xtltrain_BD_aud,xtrunclength,xtesting,grids='BD')
    print BDsvm

    xtrain_BE_aud = txmat('xtrain_BE_aud.mat','xtrain')
    ytrain_BE_aud = txmat('ytrain_BE_aud.mat','ytrain')
    xtltrain_BE_aud = txmat('xtltrain_BE_aud.mat','xtltrain')
    y,x,BEsvm = FinalClassifier.myclassify_practice_set(1,xtrain_BE_aud,ytrain_BE_aud,xtltrain_BE_aud,xtrunclength,xtesting,grids='BE')
    print BEsvm


    #math time

    finalclass = []
    # print temp1classVec.shape
    for i in range(len(audVpow)):
        elem = audVpow[i]
        if elem: #power
            temp1classVec = pow1ClassMat[:,i]
            if np.any(temp1classVec == 1):
                finalclass.append(power9Class[i])
            else:
                finalclass.append(10) #N/A
        else: #audio
            temp1classVec = aud1ClassMat[:,i]
            if np.any(temp1classVec == 1):
                if audio9Class[i] == 2: #if we guess class B, use the BD Classifier to make sure it isn't D
                    if BDsvm[i] == 2: # BDsvm ==1 is B, 2 is D
                        finalclass.append(4) # if we guessed class D but the BD svm says it's B, append B
                    else:
                        finalclass.append(audio9Class[i])
                elif audio9Class[i] == 5: #if we guess E, make sure it isn't B
                    if BEsvm[i] ==1:
                        finalclass.append(2)
                    else:
                        finalclass.append(audio9Class[i])
                else:
                    finalclass.append(audio9Class[i])
                # finalclass.append(audio9Class[i])
            else:
                finalclass.append(10) #N/A

    # print finalclass
    print FinalClassifier.predVec2Str(finalclass)

# lookatme()
# lookatme()
# lookatme()
# lookatme()
# lookatme()
# lookatme()
# lookatme()
lookatme()
lookatme()
lookatme()
lookatme()
lookatme()

# i'm confused why it gets worse with each iteration, i though i'm not saving anything between runnings!
# it looks like its saving the models or something, so i should look for some parameter which randomizes it?
# we should run it for audio through 18 one class classifiers, one off each recording, for the none of the above
