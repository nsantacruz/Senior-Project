import numpy as np

import NA_Classifier
from CurrentThingsNeededtoRun.Transfer_Mat_From_Matlab import txmat
import copy
import numpy as np
#
# xtesting = txmat('xtesting.mat','xtesting')
# xtrunclength = txmat('xtrunclength.mat','xtrunclength')



xtrain = []
xtltrain = []
grids = ['A','B','C','D','E','F','G','H','I']
# pow1ClassMat = np.empty([len(grids),12])


for grid in grids:
    xtrain.append(txmat('xtrain_' + grid + '_aud.mat','xtrain'))
    xtltrain.append(txmat('xtltrain_' + grid + '_aud.mat','xtltrain'))


xtraintest=copy.deepcopy(xtrain)

for i in range(len(xtrain)):
    for j in range(len(xtrain)):
        if np.array_equal(xtraintest,xtrain) == True:
            ystring,yvec = NA_Classifier.myclassify_oneclass(1, xtrain[i],xtrain[j], xtltrain[j], nuparam = .001)
            ystring1,yvec = NA_Classifier.myclassify_oneclass(1, xtrain[i],xtrain[j], xtltrain[j], nuparam = .03)
            ystring2,yvec = NA_Classifier.myclassify_oneclass(1, xtrain[i],xtrain[j], xtltrain[j], nuparam = .1)
            ystring3,yvec = NA_Classifier.myclassify_oneclass(1, xtrain[i],xtrain[j], xtltrain[j], nuparam = .3)
            ystring4,yvec = NA_Classifier.myclassify_oneclass(1, xtrain[i],xtrain[j], xtltrain[j], nuparam = .5)
            ystring5,yvec = NA_Classifier.myclassify_oneclass(1, xtrain[i],xtrain[j], xtltrain[j], nuparam = .7)


            # pow1ClassMat[i,:] = yvec
            print 'training on grid  ' + grids[i] + ' and testing on ' + grids[j] + ' power '
            print ystring
            print ystring1
            print ystring2
            print ystring3
            print ystring4
            print ystring5


