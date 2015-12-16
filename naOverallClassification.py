from Transfer_Mat_From_Matlab import txmat
import FinalClassifier
import AudiovsPower
import NA_Classifier
import numpy as np

def predVec2Str(ytest):
    gridLetters = 'ABCDEFGHIN'
    str = '  '
    if ytest != []:
        for pred in ytest:
            #remember, A corresponds to class 1
            tempInt = int(pred)
            str = str + gridLetters[int(pred)-1] if tempInt != -1 else str + '-'
            str += ' ' * 8
    return str

xtesting = txmat('xtesting.mat','xtesting')
xtrunclength = txmat('xtrunclength.mat','xtrunclength')

xtrainallpow = txmat('xtrain_all_pow.mat','xtrain')
xtltrainallpow = txmat('xtltrain_all_pow.mat','xtltrain')
ytrainallpow = txmat('ytrain_all_pow.mat','ytrain')

print "power results"
y,x,power9Class = FinalClassifier.myclassify_practice_set(1,xtrainallpow,ytrainallpow,xtltrainallpow,xtrunclength,xtesting)
print y

print power9Class
print ""

xtrainallaud = txmat('xtrain_all_aud.mat','xtrain')
xtltrainallaud = txmat('xtltrain_all_aud.mat','xtltrain')
ytrainallaud = txmat('ytrain_all_aud.mat','ytrain')

print 'audio results'
print ""

y,x,audio9Class = FinalClassifier.myclassify_practice_set(1,xtrain = xtrainallaud,ytrain = ytrainallaud,xtltrain = xtltrainallaud,xtltest = xtrunclength,xtest = xtesting)
print y
print audio9Class
print ""


ybintestallaud = txmat('ybintrain_all_aud.mat','ybintrain')
ybintestallpow = txmat('ybintrain_all_pow.mat','ybintrain')


y,audVpow = AudiovsPower.myclassify_AudPow(1,xtrainallaud,xtrainallpow,ybintestallaud,ybintestallpow,xtesting)
print 'results from binary audio power classifier '
print ""
print y
print ""
print audVpow

grids = ['A','B','C','D','E','F','G','H','I']
pow1ClassMat = np.empty([len(grids),len(audVpow)])
aud1ClassMat = np.empty([len(grids),len(audVpow)])

#POWER ONE-CLASS
xtrain = []
xtltrain = []



for grid in grids:
    xtrain.append(txmat('xtrain_' + grid + '_pow.mat','xtrain'))
    xtltrain.append(txmat('xtltrain_' + grid + '_pow.mat','xtltrain'))


for i in range(len(xtrain)):
    ystring,yvec = NA_Classifier.myclassify_oneclass(1,xtrain[i],xtesting,xtrunclength,nuparam = .1)
    pow1ClassMat[i,:] = yvec
    print 'results on training set for training on ' + grids[i] + ' power '
    print ystring

#AUDIO ONE-CLASS
xtrain = []
xtltrain = []
for grid in grids:
    xtrain.append(txmat('xtrain_' + grid + '_aud.mat','xtrain'))
    xtltrain.append(txmat('xtltrain_' + grid + '_aud.mat','xtltrain'))


for i in range(len(xtrain)):
    ystring,yvec = NA_Classifier.myclassify_oneclass(1,xtrain[i],xtesting,xtrunclength,nuparam = .1)
    aud1ClassMat[i,:] = yvec
    print 'results on training set for training on ' + grids[i] + ' audio '
    print ystring


#math time

finalclass = []

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
            finalclass.append(audio9Class[i])
        else:
            finalclass.append(10) #N/A

print finalclass
