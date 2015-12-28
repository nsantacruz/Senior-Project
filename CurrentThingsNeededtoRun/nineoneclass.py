import numpy as np

import NA_Classifier
from CurrentThingsNeededtoRun.Transfer_Mat_From_Matlab import txmat

#Grid A
xtrain_A_pow = txmat('xtrain_A_pow.mat','xtrain')
xtltrain_A_pow = txmat('xtltrain_A_pow.mat','xtltrain')
#Grid B
xtrain_B_pow = txmat('xtrain_B_pow.mat','xtrain')
xtltrain_B_pow = txmat('xtltrain_B_pow.mat','xtltrain')
#Grid C
xtrain_C_pow = txmat('xtrain_C_pow.mat','xtrain')
xtltrain_C_pow = txmat('xtltrain_C_pow.mat','xtltrain')

xtrain_D_pow = txmat('xtrain_D_pow.mat','xtrain')
xtltrain_D_pow = txmat('xtltrain_D_pow.mat','xtltrain')

xtrain_E_pow = txmat('xtrain_E_pow.mat','xtrain')
xtltrain_E_pow = txmat('xtltrain_E_pow.mat','xtltrain')

xtrain_F_pow = txmat('xtrain_F_pow.mat','xtrain')
xtltrain_F_pow = txmat('xtltrain_F_pow.mat','xtltrain')

xtrain_G_pow = txmat('xtrain_G_pow.mat','xtrain')
xtltrain_G_pow = txmat('xtltrain_G_pow.mat','xtltrain')

xtrain_H_pow = txmat('xtrain_H_pow.mat','xtrain')
xtltrain_H_pow = txmat('xtltrain_H_pow.mat','xtltrain')

xtrain_I_pow = txmat('xtrain_I_pow.mat','xtrain')
xtltrain_I_pow = txmat('xtltrain_I_pow.mat','xtltrain')

xtesting = txmat('xtesting.mat','xtesting')
xtrunclength = txmat('xtrunclength.mat','xtrunclength')



xtrain = []
xtltrain = []
grids = ['A','B','C','D','E','F','G','H','I']
pow1ClassMat = np.empty([len(grids),len(audVpow)])
aud1ClassMat = np.empty([len(grids),len(audVpow)])

for grid in grids:
    xtrain.append(txmat('xtrain_' + grid + '_pow.mat','xtrain'))
    xtltrain.append(txmat('xtltrain_' + grid + '_pow.mat','xtltrain'))


for i in range(len(xtrain)):
    ystring,yvec = NA_Classifier.myclassify_oneclass(1, xtrain[i], xtesting, xtrunclength, nuparam = .1)
    pow1ClassMat[i,:] = yvec
    print 'results on training set for training on ' + grids[i] + ' power '
    print ystring
print pow1ClassMat



# for i in range(len(xtrain)):
#     for j in range(len(xtrain)):
#         ystring = NA_Classifier.myclassify_oneclass(1,xtrain[i],xtrain[j],xtltrain[j],nuparam = .1)
#         print 'results for training on ' + grids[i] + ' and testing on ' + grids[j]
#         # print ""
#         print ystring
#         # print ""

#Process finished wit
# '''for i in range(len(xtrain)):
#     ystring = NA_Classifier.myclassify_oneclass(1,xtrain[i],xtesting,xtrunclength,nuparam = .1)
#     print 'results on training set for training on ' + grids[i] + ' power '
#     print ystring
#
# #
# # for i in range(len(xtrain)):
# #         ystring = NA_Classifier.myclassify_oneclass(1,xtrain[i],xtrain[j],xtltrain[j],nuparam = .1)
# #         print 'results for training on ' + grids[i] + ' and testing on ' + grids[j]
# #         # print ""
# #         print ystring
# #         # print ""
# #
#
# print ""
# print ""
# print "now for audio"
#
#
#
# #Grid A
# xtrain_A_aud = txmat('xtrain_A_aud.mat','xtrain')
# xtltrain_A_aud = txmat('xtltrain_A_aud.mat','xtltrain')
# #Grid B
# xtrain_B_aud = txmat('xtrain_B_aud.mat','xtrain')
# xtltrain_B_aud = txmat('xtltrain_B_aud.mat','xtltrain')
# #Grid C
# xtrain_C_aud = txmat('xtrain_C_aud.mat','xtrain')
# xtltrain_C_aud = txmat('xtltrain_C_aud.mat','xtltrain')
#
# xtrain_D_aud = txmat('xtrain_D_aud.mat','xtrain')
# xtltrain_D_aud = txmat('xtltrain_D_aud.mat','xtltrain')
#
# xtrain_E_aud = txmat('xtrain_E_aud.mat','xtrain')
# xtltrain_E_aud = txmat('xtltrain_E_aud.mat','xtltrain')
#
# xtrain_F_aud = txmat('xtrain_F_aud.mat','xtrain')
# xtltrain_F_aud = txmat('xtltrain_F_aud.mat','xtltrain')
#
# xtrain_G_aud = txmat('xtrain_G_aud.mat','xtrain')
# xtltrain_G_aud = txmat('xtltrain_G_aud.mat','xtltrain')
#
# xtrain_H_aud = txmat('xtrain_H_aud.mat','xtrain')
# xtltrain_H_aud = txmat('xtltrain_H_aud.mat','xtltrain')
#
# xtrain_I_aud = txmat('xtrain_I_aud.mat','xtrain')
# xtltrain_I_aud = txmat('xtltrain_I_aud.mat','xtltrain')
#
# xtesting = txmat('xtesting.mat','xtesting')
# xtrunclength = txmat('xtrunclength.mat','xtrunclength')
#
#
# xtrain = [xtrain_A_aud,xtrain_B_aud,xtrain_C_aud,xtrain_D_aud,xtrain_E_aud,xtrain_F_aud,xtrain_G_aud,xtrain_H_aud,xtrain_I_aud]
# xtltrain = [xtltrain_A_aud,xtltrain_B_aud,xtltrain_C_aud,xtltrain_D_aud,xtltrain_E_aud,xtltrain_F_aud,xtltrain_G_aud,xtltrain_H_aud,xtltrain_I_aud]
# grids = ['A','B','C','D','E','F','G','H','I']
#
#
# # for i in range(len(xtrain)):
# #     for j in range(len(xtrain)):
# #         ystring = NA_Classifier.myclassify_oneclass(1,xtrain[i],xtrain[j],xtltrain[j],nuparam = .1)
# #         print 'results for training on ' + grids[i] + ' and testing on ' + grids[j]
# #         # print ""
# #         print ystring
# #         # print ""
#
# for i in range(len(xtrain)):
#     ystring = NA_Classifier.myclassify_oneclass(1,xtrain[i],xtesting,xtrunclength,nuparam = .1)
#     print 'results on training set for training on ' + grids[i] + ' audio '
#     print ystring
#
# #
# # for i in range(len(xtrain)):
# #         ystring = NA_Classifier.myclassify_oneclass(1,xtrain[i],xtrain[j],xtltrain[j],nuparam = .1)
# #         print 'results for training on ' + grids[i] + ' and testing on ' + grids[j]
# #         # print ""
# #         print ystring
# #         # print ""
# #'''
